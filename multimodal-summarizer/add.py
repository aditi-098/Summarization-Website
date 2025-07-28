import os
import tempfile
import streamlit as st
import whisper
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, ImageClip
from gtts import gTTS
from PIL import Image, ImageDraw

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# -------------------- Configuration --------------------

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

llm_groq = ChatGroq(model_name="llama3-70b-8192")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm_gemini = GoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

# -------------------- Summarization Logic --------------------
def summarize_documents(docs, llm):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    split_docs = splitter.split_documents(docs)

    map_prompt = PromptTemplate.from_template("""
    You are an expert summarizer. Given the following document chunk, write a detailed and complete summary:
    "{text}"
    Summary:
    """)

    combine_prompt = PromptTemplate.from_template("""
    You are given a series of chunk summaries. Combine them into a single comprehensive summary that is roughly 40-50% the length of the original:
    "{text}"
    Final Summary:
    """)

    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
    return chain.run(split_docs)

def summarize_text(text, llm):
    return summarize_documents([Document(page_content=text)], llm)

def summarize_from_url(url, llm):
    return summarize_documents(WebBaseLoader(url).load(), llm)

def summarize_from_pdf(file_path, llm):
    return summarize_documents(PyPDFLoader(file_path).load(), llm)

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    return model.transcribe(audio_path)["text"]

def summarize_audio(audio_path, llm):
    transcript = transcribe_audio(audio_path)
    return summarize_text(transcript, llm), transcript

def summarize_video(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name

    video = mp.VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".mp3")
    video.audio.write_audiofile(audio_path)
    transcript = transcribe_audio(audio_path)
    summary = summarize_text(transcript, llm_groq)
    video.close()
    os.remove(video_path)
    os.remove(audio_path)
    return transcript, summary

def text_to_speech(text, output="summary_audio.mp3"):
    gTTS(text).save(output)
    return output

def create_summary_video(text, audio_path, output="summary_video.mp4"):
    img = Image.new('RGB', (1280, 720), color='white')
    draw = ImageDraw.Draw(img)
    y = 50
    for line in text.split('. '):
        draw.text((50, y), line.strip(), fill='black')
        y += 30

    img.save("summary_image.png")
    audio_clip = AudioFileClip(audio_path)
    video_clip = ImageClip("summary_image.png").set_duration(audio_clip.duration).set_audio(audio_clip)
    video_clip.write_videofile(output, fps=24)
    return output

# -------------------- Streamlit UI --------------------
st.set_page_config(layout="wide")
st.title("\U0001F4CA Multimodal Summarization")

llm_choice = st.sidebar.selectbox("Choose LLM Backend", ["Groq", "Gemini"])
llm = llm_groq if llm_choice == "Groq" else llm_gemini

page = st.sidebar.radio("Select Task", ["Text Summarization", "Audio Summarization", "Video Summarization"])

if page == "Text Summarization":
    st.header("Text Summarization")
    method = st.radio("Choose Input Type", ["Plain Text", "PDF File", "URL"])

    if method == "Plain Text":
        text = st.text_area("Enter Text")
        if st.button("Summarize Text") and text:
            summary = summarize_text(text, llm)
            st.success("Summary")
            st.write(summary)

    elif method == "PDF File":
        uploaded = st.file_uploader("Upload PDF", type="pdf")
        if uploaded and st.button("Summarize PDF"):
            path = f"temp_{uploaded.name}"
            with open(path, "wb") as f:
                f.write(uploaded.read())
            summary = summarize_from_pdf(path, llm)
            st.success("Summary")
            st.write(summary)
            os.remove(path)

    elif method == "URL":
        url = st.text_input("Enter URL")
        if st.button("Summarize URL") and url:
            summary = summarize_from_url(url, llm)
            st.success("Summary")
            st.write(summary)

elif page == "Audio Summarization":
    st.header("Audio Summarization")
    audio = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    if audio and st.button("Summarize Audio"):
        path = f"temp_{audio.name}"
        with open(path, "wb") as f:
            f.write(audio.read())
        summary, transcript = summarize_audio(path, llm)
        st.subheader("Transcript")
        st.write(transcript)
        st.subheader("Summary")
        st.success(summary)
        audio_out = text_to_speech(summary)
        st.audio(audio_out)
        os.remove(path)

elif page == "Video Summarization":
    st.header("Video Summarization")
    video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if video and st.button("Summarize Video"):
        transcript, summary = summarize_video(video)
        st.subheader("Transcript")
        st.write(transcript)
        st.subheader("Summary")
        st.success(summary)
        audio_file = text_to_speech(summary)
        summary_video = create_summary_video(summary, audio_file)
        st.video(summary_video)
        st.download_button("Download Summary Video", data=open(summary_video, "rb"), file_name="summary_video.mp4")
