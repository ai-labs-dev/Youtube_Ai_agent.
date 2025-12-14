import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="YouTube AI Summarizer", layout="wide")

# Sidebar for API Key
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("groq_api_key", type="password")

# --- FUNCTIONS ---

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    Examples:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID?si=...
    """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be" in url:
        # This fixes the issue by removing the "?si=..." tracker
        return url.split("/")[-1].split("?")[0]
    return None

def get_transcript(video_id):
    """
    Fetches the transcript (subtitles) of the video using the YouTube API.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine all parts of the transcript into one big string
        full_text = " ".join([item['text'] for item in transcript_list])
        return full_text
    except Exception as e:
        return None

def generate_summary(text, api_key):
    """
    Uses Groq AI to summarize the transcript.
    """
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-8b-8192")
    
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert video summarizer.
        Read the following transcript from a YouTube video and provide a concise, structured summary.
        Capture the main points, key takeaways, and any actionable advice.
        
        Transcript: {text}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({"text": text})
    return response.content

def ask_question(text, question, api_key):
    """
    Uses Groq AI to answer a specific question based on the transcript.
    """
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama3-8b-8192")
    
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question based ONLY on the following video transcript.
        If the answer is not in the transcript, say "I couldn't find that in the video."
        
        Transcript: {text}
        
        Question: {question}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({"text": text, "question": question})
    return response.content

# --- FRONTEND UI ---

st.title("📺 YouTube Video AI Agent")
st.markdown("Summarize videos, read subtitles, and chat with content instantly.")

# Input Section
video_url = st.text_input("Paste YouTube Video Link here:")

if video_url and api_key:
    video_id = extract_video_id(video_url)
    
    if video_id:
        # Display Video
        st.video(video_url)
        
        # Load Transcript
        with st.spinner("Fetching transcript..."):
            transcript_text = get_transcript(video_id)
        
        if transcript_text:
            # Create Tabs for neat organization
            tab1, tab2, tab3 = st.tabs(["📝 Summary", "💬 Chat with Video", "📜 Full Transcript"])
            
            # Tab 1: Summary
            with tab1:
                if st.button("Generate Summary"):
                    with st.spinner("Analyzing video content..."):
                        summary = generate_summary(transcript_text, api_key)
                        st.markdown(summary)
            
            # Tab 2: Chat / Q&A
            with tab2:
                st.info("Ask any question about the video content!")
                user_question = st.text_input("Ask a question:")
                if user_question:
                    with st.spinner("Thinking..."):
                        answer = ask_question(transcript_text, user_question, api_key)
                        st.write(answer)

            # Tab 3: Subtitles
            with tab3:
                st.text_area("Subtitles", transcript_text, height=300)
                
        else:
            st.error("Could not retrieve transcript. Note: This tool only works on videos that have Closed Captions/Subtitles enabled.")
    else:
        st.error("Invalid YouTube URL.")
elif not api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to proceed.")
