import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain_openai.llms import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
import openai
import whisper
import os

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Create a collection in ChromaDB
collection = chroma_client.create_collection("youtube_video_transcripts2")

# Initialize the SentenceTransformer model for embedding text
sentence_model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Whisper model for speech recognition
whisper_model = whisper.load_model("base")

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"

# LangChain Prompt Template for QA
QA_TEMPLATE = """
Use the following context from a YouTube video transcript to answer the question:

Context: {context}

Question: {question}

Answer:
"""
qa_prompt = PromptTemplate(input_variables=["context", "question"], template=QA_TEMPLATE)

# Initialize the LLM (using GPT-4 mini model)
llm = ChatOpenAI(model="gpt-4-mini", temperature=0)

def get_video_transcript(video_id: str) -> str:
    """
    Fetch transcript for a given YouTube video ID.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([item['text'] for item in transcript if 'text' in item])
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript for video {video_id}: {e}")
        return ""

def embed_text(text: str):
    """
    Convert text into embeddings using SentenceTransformer.
    """
    return sentence_model.encode([text])[0]

def upsert_to_chromadb(video_id: str, transcript_text: str):
    """
    Store video transcript embeddings in ChromaDB.
    """
    embedding = embed_text(transcript_text)
    collection.upsert(
        ids=[video_id],  # Adding the 'ids' argument with the video_id
        embeddings=[embedding],
        metadatas=[{"video_id": video_id, "transcript": transcript_text}]
    )

# Upsert all transcripts into ChromaDB
video_ids = ['2u4ItZerRac', 'I2zF1I60hPg', '8xqSF-uHCUs', 'LtmS-c1pChY']
transcripts = {}

for video_id in video_ids:
    transcript = get_video_transcript(video_id)
    if transcript:
        transcripts[video_id] = transcript
        upsert_to_chromadb(video_id, transcript)
        print(f"Stored embedding for {video_id}")
    else:
        print(f"No transcript available for video {video_id}")

def answer_question(question: str, video_id: str) -> str:
    """
    Answer a question based on the transcript of the selected video.
    """
    # Retrieve relevant video context using similarity search from ChromaDB
    results = collection.query(
        query_embeddings=[embed_text(question)],
        n_results=1
    )
    
    # Correctly access the transcript text from the results
    context_text = results['metadatas'][0]['transcript']
    
    # Create a QA chain with the OpenAI model and the context
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
    answer = qa_chain.invoke(context=context_text, question=question)
    
    return answer

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio input into text using Whisper.
    """
    result = whisper_model.transcribe(audio_path)
    return result['text']

def chatbot_interface(question: str, video_title: str, audio_file: str = None) -> str:
    """
    Handle user queries and answer questions based on selected video.
    """
    video_choices = {
        'Video 1': '2u4ItZerRac',
        'Video 2': 'I2zF1I60hPg',
        'Video 3': '8xqSF-uHCUs',
        'Video 4': 'LtmS-c1pChY'
    }
    
    video_id = video_choices.get(video_title)
    
    if audio_file:
        # Transcribe audio if provided
        question = transcribe_audio(audio_file)
    
    if video_id:
        return answer_question(question, video_id)
    else:
        return "Please select a valid video."

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[
        gr.Textbox(label="Enter your question"),
        gr.Dropdown(label="Select Video", choices=['Video 1', 'Video 2', 'Video 3', 'Video 4']),
        gr.Audio(label="Upload Audio (optional)", type="file", optional=True)
    ],
    outputs="text"
)

# Launch the Gradio app
iface.launch(share=True)

# Performance Evaluation
# Define a set of test queries and expected answers
test_cases = [
    {
        "question": "What are the benefits of immigration to Germany?",
        "expected_answer": "Immigration brings cultural diversity, fills labor shortages, and contributes to economic growth."
    },
    {
        "question": "What are the challenges faced by immigrants in Germany?",
        "expected_answer": "Immigrants may face language barriers, integration difficulties, and potential discrimination."
    },
    # Add more test cases as needed
]

def evaluate_model(test_cases):
    """
    Evaluate the model with a set of test cases.
    """
    correct_answers = 0
    for test in test_cases:
        answer = answer_question(test['question'], '2u4ItZerRac')  # Assuming using one video for testing
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected_answer']}")
        print(f"Received: {answer}")
        if test['expected_answer'].lower() in answer.lower():
            correct_answers += 1
        print()
    
    accuracy = correct_answers / len(test_cases)
    print(f"Model accuracy: {accuracy * 100}%")

# Run the evaluation
evaluate_model(test_cases)