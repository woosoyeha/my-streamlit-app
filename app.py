
import os
import json
import hashlib
import tempfile

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# ==================================================
# 1. RAG 보조 함수
# ==================================================

# 검색된 문서를 페이지 출처와 함께 문자열로 변환
def format_docs(docs):
    result = []

    for doc in docs:
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page = page + 1
        else:
            page = "미확인"

        result.append(
            f"[출처: 백서 {page}페이지]\n{doc.page_content}"
        )

    return "\n\n".join(result)


# 중복되거나 비어 있는 검색어 제거
def unique_queries(queries, limit):
    if not isinstance(queries, list):
        return []

    result = []

    for query in queries:
        if not isinstance(query, str):
            continue

        query = query.strip()

        if query and query not in result:
            result.append(query)

    return result[:limit]


# 문서 중복 확인용 키
def doc_key(doc):
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.page_content
    )


# 청크 비교를 위한 문자 5-gram
def char_ngrams(text, n=5):
    compact = "".join(text.split())

    if len(compact) < n:
        return {compact} if compact else set()

    return {
        compact[i:i+n]
        for i in range(len(compact) - n + 1)
    }


# 같은 페이지의 두 청크가 얼마나 겹치는지 계산
def overlap_ratio(doc_a, doc_b):

    source_a = doc_a.metadata.get("source")
    page_a = doc_a.metadata.get("page")

    source_b = doc_b.metadata.get("source")
    page_b = doc_b.metadata.get("page")

    if (source_a, page_a) != (source_b, page_b):
        return 0.0

    a = char_ngrams(doc_a.page_content)
    b = char_ngrams(doc_b.page_content)

    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


# 중복 비율이 높은 청크 제외
def add_if_distinct(selected, candidate, threshold=0.82):

    if any(
        overlap_ratio(candidate, existing) >= threshold
        for existing in selected
    ):
        return False

    selected.append(candidate)

    return True


# JSON 코드블록 제거
def remove_json_fences(raw):
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()

    return raw


# ==================================================
# 2. RAG 파이프라인
# ==================================================

def rag_pipeline(filepath):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "PDF에서 텍스트를 추출하지 못했습니다."
        )

    # 임베딩 모델
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 사용자별 임시 폴더에 벡터 저장소 생성
    chroma_dir = os.path.join(
        os.path.dirname(filepath),
        "chroma_db"
    )

    vectorstore = Chroma(
        collection_name="aivle_lib",
        embedding_function=embedding,
        persist_directory=chroma_dir
    )

    # 현재 PDF를 벡터 저장소에 추가
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)

    # MMR 검색기
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # BM25 검색기
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 6

    # 하이브리드 검색기
    retriever = EnsembleRetriever(
        retrievers=[mmr_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # ----------------------------------------------
    # 질문 분류 및 검색 질문 생성
    # ----------------------------------------------

    query_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 AIVLE School 백서의 검색 질문 설계자입니다.

사용자 질문을 단순(simple) 또는 복합(complex)으로 구분하세요.

- 단순: 하나의 사실, 규정 또는 조건을 확인하는 질문
- 복합: 서로 다른 사실이나 판단 항목을 2개 이상 함께 묻는 질문

[규칙]

1. 사용자 질문을 simple 또는 complex로 분류하세요.

2. 단순 질문은 multi_queries를 정확히 2개 생성하세요.

3. 복합 질문은 multi_queries를 정확히 5개 생성하세요.

4. multi_queries에 원본 질문과 동일한 문장을 포함하지 마세요.
   생성된 질문끼리도 중복되지 않아야 합니다.

5. 서로 다른 검색 표현과 관점을 사용하세요.

- 핵심 개념 중심
- 관련 규정 및 조건 중심
- 관련 절차 및 방법 중심
- 동의어 및 유사 표현 활용
- 질문의 세부 항목 중심

원본 질문과 관련 없는 내용을 추가하지 마세요.

6. sub_queries는 원본 질문에서 답해야 할 독립적인 항목으로 분리하세요.

- 단순 질문: 정확히 1개
- 복합 질문: 최대 4개
- 원본 질문의 모든 요구사항을 포함하세요.
- 독립 항목이 4개를 초과하면 관련 항목을 묶으세요.

7. 원본 질문의 기수, 날짜, 수치, 조건은 변경하거나 누락하지 마세요.

8. 문서에 없는 사실을 전제하지 말고 검색 질문만 생성하세요.

9. JSON 객체 하나만 출력하세요.
   코드블록이나 부연 설명은 작성하지 마세요.

[출력 형식]

{{
    "complexity": "simple 또는 complex",
    "multi_queries": ["..."],
    "sub_queries": ["..."]
}}
"""
        ),
        ("human", "{input}")
    ])

    query_chain = (
        query_prompt
        | llm
        | StrOutputParser()
    )

    # 검색 질문 생성
    def generate_queries(question):

        try:
            raw = query_chain.invoke(
                {"input": question}
            )

            raw = remove_json_fences(raw)
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError(
                    "질문 생성 결과가 JSON 객체가 아닙니다."
                )

            complexity = data.get(
                "complexity",
                "simple"
            )

            if complexity not in ["simple", "complex"]:
                complexity = "simple"

            if complexity == "complex":
                multi_limit = 5
                sub_limit = 4
            else:
                multi_limit = 2
                sub_limit = 1

            multi_queries = unique_queries(
                data.get("multi_queries", []),
                multi_limit
            )

            sub_queries = unique_queries(
                data.get("sub_queries", []),
                sub_limit
            )

            if len(multi_queries) != multi_limit:
                print(
                    f"[Multi-Query 생성 개수 부족: "
                    f"목표 {multi_limit}개 / "
                    f"실제 {len(multi_queries)}개]"
                )

        except (ValueError, TypeError, KeyError) as exc:

            print(
                f"[질문 생성 형식 오류, "
                f"원문 검색으로 대체: {exc}]"
            )

            complexity = "simple"
            multi_queries = []
            sub_queries = [question]

        if not sub_queries:
            sub_queries = [question]

        return complexity, multi_queries, sub_queries

    # ----------------------------------------------
    # Multi-Query 및 Sub-Query 검색
    # ----------------------------------------------

    def multi_retrieve(question):

        complexity, multi_queries, sub_queries = (
            generate_queries(question)
        )

        # 원문 질문을 먼저 검색
        queries = unique_queries(
            [question] + multi_queries + sub_queries,
            10
        )

        print(
            f"[질문 유형: {complexity} / "
            f"Multi-Query {len(multi_queries)}개 / "
            f"Sub-Query {len(sub_queries)}개]"
        )

        print("[Multi-Query]", multi_queries)
        print("[Sub-Query]", sub_queries)

        doc_map = {}
        scores = {}
        results_by_query = {}

        for query_index, query in enumerate(queries):

            found_docs = retriever.invoke(query)
            results_by_query[query] = found_docs

            for rank, doc in enumerate(found_docs):

                key = doc_key(doc)
                doc_map[key] = doc

                # 원본 질문 검색 결과에 가중치 부여
                if query_index == 0:
                    weight = 1.5
                else:
                    weight = 1.0

                scores[key] = (
                    scores.get(key, 0.0)
                    + weight / (60 + rank + 1)
                )

        selected = []

        # 각 Sub-Query의 관련 문서를 우선 확보
        for sub_query in sub_queries:

            for candidate in results_by_query.get(
                sub_query,
                []
            ):

                if add_if_distinct(selected, candidate):
                    break

            if len(selected) >= 10:
                break

        # 나머지는 점수가 높은 순서대로 추가
        for key in sorted(
            scores,
            key=scores.get,
            reverse=True
        ):

            if len(selected) >= 10:
                break

            add_if_distinct(
                selected,
                doc_map[key]
            )

        print(
            f"[정확 중복 제거 후 {len(doc_map)}개 / "
            f"최종 선택 {len(selected)}개]"
        )

        return selected

    # ----------------------------------------------
    # 답변 생성
    # ----------------------------------------------

    s_msg = """당신은 AIVLE School 백서 기반 학습 어시스턴트입니다.

1. 제공된 문서(context)에서 확인한 사실만 근거로 답변하세요.

2. 시간, 금액, 조건, 절차는 문서에 있으면 구체적으로 설명하세요.

3. '다른 항목 참조'라는 답변만 하지 말고 실제 내용을 설명하세요.

4. 복합 질문은 질문별로 나누어 빠짐없이 답하세요.

5. 없는 정보는 추측하지 말고
   '제공된 문서에서 확인되지 않습니다'라고 안내하세요.

6. 기수별 기준 충돌이나 확인 필요 사항은 확정하지 마세요.

7. 선배 개인 경험을 공식 의무로 해석하지 마세요.

8. 답변의 각 주요 주장에 이를 뒷받침하는 출처 페이지를 붙이세요.
   문서에 없는 페이지는 만들어내지 마세요.

[검색된 문서]
{context}
"""

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg),
        ("human", "{input}")
    ])

    answer_chain = (
        answer_prompt
        | llm
        | StrOutputParser()
    )

    # ----------------------------------------------
    # 답변 검증
    # ----------------------------------------------

    verification_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """당신은 백서 기반 RAG 답변의 엄격한 검증자입니다.

원본 질문, 검색된 문서, 초안 답변을 비교하여 검증하세요.

1. 질문 의도: 질문의 핵심 요구에 답했는가?

2. 답변 완전성: 독립적인 질문 항목을 빠짐없이 다뤘는가?

3. 문서 근거: 사실, 숫자, 조건, 출처 페이지가
   검색된 문서와 일치하는가?

4. 정보 부재: 근거가 없는 내용은 추측하지 않고
   확인되지 않는다고 명시했는가?

[주의사항]

검색 문서에 답이 없다면 모른다고 안내하는 것은 올바른 답변입니다.

검색 문서에 없는 정보를 반드시 답하라고 요구하지 마세요.

사소한 문체 문제만으로 실패시키지 마세요.

아래 JSON 객체만 출력하세요.
코드블록은 쓰지 마세요.

{{"pass": true, "feedback": "검증 통과"}}

또는

{{"pass": false, "feedback": "부족한 내용과 수정 방법"}}
"""
        ),
        (
            "human",
            """원본 질문:
{input}

검색된 문서:
{context}

초안 답변:
{answer}"""
        )
    ])

    verification_chain = (
        verification_prompt
        | llm
        | StrOutputParser()
    )

    # ----------------------------------------------
    # 검증 실패 시 답변 재생성
    # ----------------------------------------------

    retry_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            s_msg + """

[재생성 지침]

초안에서 검증 실패한 부분을 바로잡아 전체 답변을 다시 작성하세요.

검증 피드백을 반영하되,
백서의 근거보다 피드백을 우선하지 마세요.

검색된 문서에서 확인할 수 없는 내용은
추측하지 말고 미확인으로 명시하세요.

이 단계에서는 다시 검증하거나 재생성하지 않습니다.
"""
        ),
        (
            "human",
            """원본 질문:
{input}

이전 답변:
{answer}

검증 피드백:
{feedback}"""
        )
    ])

    retry_chain = (
        retry_prompt
        | llm
        | StrOutputParser()
    )

    # ----------------------------------------------
    # 검색 → 답변 → 검증 → 필요 시 재생성
    # ----------------------------------------------

    def answer_with_verification(payload):

        question = payload["input"]

        # 검색된 문서를 초안과 검증에 동일하게 사용
        selected_docs = multi_retrieve(question)
        context = format_docs(selected_docs)

        values = {
            "input": question,
            "context": context
        }

        # 초안 생성
        answer = answer_chain.invoke(values)

        try:
            # 초안 검증
            raw_verdict = verification_chain.invoke(
                {
                    **values,
                    "answer": answer
                }
            )

            raw_verdict = remove_json_fences(
                raw_verdict
            )

            verdict = json.loads(raw_verdict)

            if (
                not isinstance(verdict, dict)
                or type(verdict.get("pass")) is not bool
            ):
                raise ValueError(
                    "검증 결과에 불리언 pass가 없습니다."
                )

            passed = verdict["pass"]

            feedback = str(
                verdict.get("feedback", "")
            ).strip()

            if not passed and not feedback:
                feedback = (
                    "질문 의도와 문서 근거를 다시 확인하여 "
                    "누락을 보완하세요."
                )

        except (ValueError, TypeError) as exc:

            passed = False

            feedback = (
                "검증 결과를 해석하지 못했습니다. "
                "원본 질문과 검색 문서를 다시 비교해 "
                "답변을 보완하세요."
            )

            print(f"[검증 형식 오류: {exc}]")

        # 검증 통과
        if passed:
            print("[답변 검증: 통과 / 재생성 0회]")
            return answer

        # 검증 실패
        print(
            f"[답변 검증: 실패 / "
            f"피드백: {feedback}]"
        )

        revised = retry_chain.invoke(
            {
                **values,
                "answer": answer,
                "feedback": feedback
            }
        )

        print(
            "[답변 재생성 1회 완료 / "
            "추가 검증 없음]"
        )

        return revised

    return RunnableLambda(
        answer_with_verification
    )


# ==================================================
# 3. Streamlit 화면
# ==================================================

st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘"
)

st.title("📘 AIVLE School 학습 도우미")

st.write(
    "AIVLE School 백서 PDF를 업로드하고 "
    "궁금한 내용을 질문해 주세요."
)


# OpenAI API 키 확인
if not os.environ.get("OPENAI_API_KEY"):

    try:
        os.environ["OPENAI_API_KEY"] = (
            st.secrets["OPENAI_API_KEY"]
        )

    except (KeyError, FileNotFoundError):

        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "Streamlit Secrets에 OPENAI_API_KEY를 등록해 주세요."
        )

        st.stop()


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None


# PDF 업로드
uploaded_file = st.file_uploader(
    "백서 PDF를 업로드해 주세요.",
    type=["pdf"]
)


# 대화 삭제 버튼
if st.button("🗑️ 대화 내용 지우기"):
    st.session_state.messages = []
    st.rerun()


# PDF 업로드 및 RAG 파이프라인 생성
if uploaded_file is not None:

    # PDF 내용 읽기
    pdf_bytes = uploaded_file.getvalue()

    # 동일한 PDF인지 확인할 해시값 생성
    current_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    # 새로운 PDF를 업로드했을 때만 파이프라인 생성
    if current_hash != st.session_state.pdf_hash:

        with st.spinner(
            "PDF를 읽고 검색기를 만들고 있습니다. "
            "잠시 기다려 주세요."
        ):

            temp_dir = tempfile.TemporaryDirectory()

            try:
                # 임시 폴더에 PDF 저장
                pdf_path = os.path.join(
                    temp_dir.name,
                    "aivle_whitepaper.pdf"
                )

                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                # 위에서 만든 RAG 파이프라인 실행
                new_chain = rag_pipeline(pdf_path)

                # 이전 임시 폴더 보관
                old_temp_dir = st.session_state.temp_dir

                # 새 파이프라인과 PDF 정보 저장
                st.session_state.rag_chain = new_chain
                st.session_state.pdf_hash = current_hash
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.temp_dir = temp_dir

                # PDF가 바뀌면 대화 기록 초기화
                st.session_state.messages = []

                # 이전 임시 폴더 정리
                if old_temp_dir is not None:
                    old_temp_dir.cleanup()

                st.success(
                    f"'{uploaded_file.name}' "
                    "파일 분석이 완료되었습니다."
                )

            except Exception as e:

                temp_dir.cleanup()

                st.error(
                    "PDF를 처리하는 중 오류가 발생했습니다."
                )

                st.exception(e)

                st.stop()

    else:
        st.success(
            f"현재 사용 중인 파일: "
            f"{st.session_state.pdf_name}"
        )

else:
    st.info("먼저 백서 PDF를 업로드해 주세요.")


# 이전 대화 출력
for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        role = "user"
    else:
        role = "assistant"

    with st.chat_message(role):
        st.markdown(message.content)


# 사용자 질문 입력
user_input = st.chat_input(
    "AIVLE School에 대해 궁금한 내용을 입력하세요."
)


if user_input:

    # PDF를 아직 처리하지 않은 경우
    if st.session_state.rag_chain is None:

        st.warning(
            "먼저 백서 PDF를 업로드해 주세요."
        )

    else:

        # 사용자 질문 표시
        with st.chat_message("user"):
            st.markdown(user_input)

        # 사용자 질문 저장
        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )

        # 답변 생성
        with st.chat_message("assistant"):

            with st.spinner(
                "백서에서 내용을 찾고 "
                "답변을 검증하고 있습니다..."
            ):

                try:
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input}
                    )

                    st.markdown(response)

                    # 답변 저장
                    st.session_state.messages.append(
                        AIMessage(content=response)
                    )

                except Exception as e:

                    st.error(
                        "답변을 생성하는 중 오류가 발생했습니다."
                    )

                    st.exception(e)
