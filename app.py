
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# RAG 파이프라인: 하이브리드 검색 + 질문 적응형 Multi-Query + Sub-Query 근거 보존
import json

def rag_pipeline(filepath):
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="aivle_lib",
        embedding_function=embedding,
        persist_directory="chroma_db"
    )
    # 기존 동작 유지: 실행할 때 컬렉션을 초기화하고 현재 PDF를 넣음
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6
    retriever = EnsembleRetriever(
        retrievers=[mmr_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 개선 ①·②: 한 번의 LLM 호출에서 복잡도 판단, Multi-Query, Sub-Query 생성
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 AIVLE School 백서의 검색 질문 설계자입니다.
      사용자 질문을 먼저 단순(simple) 또는 복합(complex)으로 구분하세요.
      - 단순: 하나의 사실/규정/조건을 확인하는 질문
      - 복합: 서로 다른 사실이나 판단 항목을 2개 이상 함께 묻는 질문

      규칙:
      규칙:
      1. 사용자 질문을 단순(simple) 또는 복합(complex)으로 분류하세요.

      2. 단순 질문은 multi_queries를 정확히 2개 생성하세요.

      3. 복합 질문은 multi_queries를 정확히 5개 생성하세요.

      4. multi_queries는 원본 질문과 동일한 문장을 포함하지 마세요.
        생성된 질문끼리도 중복되지 않아야 합니다.

      5. multi_queries는 서로 다른 검색 표현과 관점을 사용하세요.
        - 핵심 개념 중심
        - 관련 규정 및 조건 중심
        - 관련 절차 및 방법 중심
        - 동의어 및 유사 표현 활용
        - 질문의 세부 항목 중심
        단, 원본 질문과 관련 없는 내용을 추가하지 마세요.

      6. sub_queries는 원본 질문에서 답해야 할 독립적인 항목으로 분리하세요.
        - 단순 질문: 정확히 1개
        - 복합 질문: 최대 4개
        - 원본 질문의 모든 요구사항을 포함하세요.
        - 독립 항목이 4개를 초과하면 관련 항목을 묶어 구성하세요.

      7. 원본 질문에 등장한 기수, 날짜, 수치, 조건은 변경하거나 누락하지 마세요.

      8. 문서에 없는 사실을 전제하지 말고 검색을 위한 질문만 생성하세요.

      9. 반드시 지정된 개수의 multi_queries를 포함한 JSON 객체 하나만 출력하세요.
        코드블록이나 부연 설명은 작성하지 마세요.

      출력 형식:
      {{"complexity":"simple 또는 complex",
        "multi_queries":["..."],
        "sub_queries":["..."]}}
      {{"complexity":"simple 또는 complex", "multi_queries":["..."], "sub_queries":["..."]}}"""),
              ("human", "{input}")
          ])
    query_chain = query_prompt | llm | StrOutputParser()



    def generate_queries(question):
        try:
            raw = query_chain.invoke({"input": question}).strip()
            # 혹시 모델이 JSON 코드펜스를 출력한 경우 처리
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            complexity = data.get("complexity", "simple")
            complexity = complexity if complexity in ("simple", "complex") else "simple"
            limit = 5 if complexity == "complex" else 2
            multi_queries = unique_queries(data.get("multi_queries", []), limit)
            if len(multi_queries) != limit:
              print(
                  f"[Multi-Query 생성 개수 부족: "
                  f"목표 {limit}개 / 실제 {len(multi_queries)}개]"
              )
            sub_queries = unique_queries(data.get("sub_queries", []), 4 if complexity == "complex" else 1)
        except (ValueError, TypeError, KeyError) as exc:
            print(f"[질문 생성 형식 오류, 원문 검색으로 대체: {exc}]")
            complexity, multi_queries, sub_queries = "simple", [], [question]
        if not sub_queries:
            sub_queries = [question]
        return complexity, multi_queries, sub_queries

    def multi_retrieve(question):
        complexity, multi_queries, sub_queries = generate_queries(question)
        # 원문을 먼저 검색하고, 이어서 Multi-Query / Sub-Query 검색
        queries = unique_queries([question] + multi_queries + sub_queries, 10)
        print(f"\n[질문 유형: {complexity} / Multi-Query {len(multi_queries)}개 / Sub-Query {len(sub_queries)}개]")
        print("[Multi-Query]", multi_queries)
        print("[Sub-Query]", sub_queries)

        # 검색 결과 점수 통합 + 하위 질문별 검색 결과 별도 보관
        doc_map, scores = {}, {}
        results_by_query = {}
        for qi, q in enumerate(queries):
            found = retriever.invoke(q)
            results_by_query[q] = found
            for rank, doc in enumerate(found):
                key = doc_key(doc)
                doc_map[key] = doc
                # reciprocal-rank 형태의 통합 점수. 원문 질문에 소폭 가중치
                scores[key] = scores.get(key, 0.0) + (1.5 if qi == 0 else 1.0) / (60 + rank + 1)

        # 개선 ③: 각 Sub-Query 관련 문서 1개씩 먼저 확보
        selected = []
        for sub_q in sub_queries:
            for candidate in results_by_query.get(sub_q, []):
                if add_if_distinct(selected, candidate):
                    break
            if len(selected) >= 10:
                break

        # 개선 ④: 나머지는 통합 점수 상위 순으로, 같은 페이지의 과도한 중복 제외
        for key in sorted(scores, key=scores.get, reverse=True):
            if len(selected) >= 10:
                break
            add_if_distinct(selected, doc_map[key])

        print(f"[정확 중복 제거 후 {len(doc_map)}개 / 최종 선택 {len(selected)}개]")
        return selected

    s_msg = """당신은 AIVLE School 백서 기반 학습 어시스턴트입니다.
1. 제공된 문서(context)에서 확인한 사실만 근거로 답변하세요.
2. 시간, 금액, 조건, 절차는 문서에 있으면 구체적으로 설명하세요.
3. '다른 항목 참조'라는 답변만 하지 말고 실제 내용을 설명하세요.
4. 복합 질문은 질문별로 나누어 빠짐없이 답하세요.
5. 없는 정보는 추측하지 말고 '제공된 문서에서 확인되지 않습니다'라고 하세요.
6. 기수별 기준 충돌이나 확인 필요 사항은 확정하지 마세요.
7. 선배 개인 경험을 공식 의무로 해석하지 마세요.
8. 답변의 각 주요 주장에 이를 뒷받침하는 출처 페이지를 붙이세요.
   문서에 없는 페이지는 만들어내지 마세요.

[검색된 문서]
{context}"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg), ("human", "{input}")
    ])
    # 개선 ⑤: 원본 질문과 초안의 의도·완전성·근거를 검증하는 별도 체인
    verification_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 백서 기반 RAG 답변의 엄격한 검증자입니다.
원본 질문, 검색된 문서, 초안 답변을 비교해 다음을 검증합니다.
1. 질문 의도: 질문의 핵심 요구에 답했는가?
2. 답변 완전성: 질문의 모든 독립 항목을 빠짐없이 다뤘는가?
3. 문서 근거: 초안의 사실·숫자·조건·출처 페이지가 검색 문서와 일치하는가?
4. 정보 부재: 근거가 없는 항목을 추측하지 않고 미확인으로 명시했는가?

주의: 검색 문서에 답이 없다면, 모른다고 명시하는 것은 올바른 답변입니다.
검색 문서에 없는 정보를 반드시 답하라고 요구하지 마세요.
사소한 문체 문제만으로 실패시키지 마세요.
아래 JSON 객체만 출력하세요. 코드블록은 쓰지 마세요.
{{"pass":true,"feedback":"검증 통과"}} 또는
{{"pass":false,"feedback":"어떤 항목이 왜 부족하며 어떻게 고칠지 구체적으로 설명"}}"""),
        ("human", "원본 질문:\n{input}\n\n검색된 문서:\n{context}\n\n초안 답변:\n{answer}")
    ])
    # 주의: 위 JSON 예시는 중괄호를 이스케이프해 프롬프트 변수로 오인하지 않게 함
    verification_chain = verification_prompt | llm | StrOutputParser()

    retry_prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg + """

[재생성 지침]
초안에서 검증 실패한 부분만 바로잡아 전체 답변을 다시 작성하세요.
검증 피드백을 반영하되, 백서 근거보다 피드백을 우선하지 마세요.
검색 문서에서 확인할 수 없는 정보는 추측하지 말고 미확인으로 명시하세요.
이 단계에서는 다시 검증하거나 재생성하지 않습니다."""),
        ("human", "원본 질문:\n{input}\n\n이전 답변:\n{answer}\n\n검증 피드백:\n{feedback}")
    ])
    retry_chain = retry_prompt | llm | StrOutputParser()
    answer_chain = prompt | llm | StrOutputParser()

    def remove_json_fences(raw):
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw

    def answer_with_verification(payload):
        question = payload["input"]
        # 검색은 1회만 진행하고 초안·검증·재생성에서 같은 문서를 사용
        selected_docs = multi_retrieve(question)
        context = format_docs(selected_docs)
        values = {"input": question, "context": context}
        answer = answer_chain.invoke(values)

        try:
            raw = verification_chain.invoke({**values, "answer": answer})
            verdict = json.loads(remove_json_fences(raw))
            if not isinstance(verdict, dict) or type(verdict.get("pass")) is not bool:
                raise ValueError("검증 JSON에 불리언 pass가 없습니다")
            passed = verdict["pass"]
            feedback = str(verdict.get("feedback", "")).strip()
            if not passed and not feedback:
                feedback = "질문 의도와 문서 근거를 다시 확인하여 누락을 보완하세요."
        except (ValueError, TypeError) as exc:
            # 검증 형식이 잘못되면 안전하게 1회 수정 시도 (무한 반복 방지)
            passed = False
            feedback = "검증 결과를 해석하지 못했습니다. 원본 질문과 검색 문서를 다시 비교해 답변을 보완하세요."
            print(f"[검증 형식 오류: {exc}]")

        if passed:
            print("[답변 검증: 통과 / 재생성 0회]")
            return answer

        print(f"[답변 검증: 실패 / 피드백: {feedback}]")
        revised = retry_chain.invoke({**values, "answer": answer, "feedback": feedback})
        print("[피드백 반영 답변 재생성: 1회 완료 / 추가 검증 없음]")
        return revised

    # 기존 rag_chain.invoke({"input": ...}) 인터페이스 유지
    return RunnableLambda(answer_with_verification)


# 페이지 설정
st.set_page_config(page_title="AIVLE School 학습 도우미", page_icon="📘")
st.title("📘 AIVLE School 학습 도우미")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 출력
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# 사용자 질문 입력
user_input = st.chat_input("메시지를 입력하세요.")

if user_input:

    if st.session_state.rag_chain is None:
        st.warning("먼저 백서 PDF를 업로드해 주세요.")

    else:
        # 사용자 질문 표시
        with st.chat_message("user"):
            st.markdown(user_input)

        # 사용자 질문 저장
        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )

        # 챗봇 답변 생성
        with st.chat_message("assistant"):

            with st.spinner(
                "백서에서 내용을 찾고 답변을 검증하고 있습니다..."
            ):

                try:
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input}
                    )

                    st.markdown(response)

                    # 챗봇 답변 저장
                    st.session_state.messages.append(
                        AIMessage(content=response)
                    )

                except Exception as e:
                    st.error("답변을 생성하는 중 오류가 발생했습니다.")
                    st.exception(e)
