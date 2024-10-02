import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
import cv2
import numpy as np
import threading

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize speech recognition and text-to-speech engines
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return listen()
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return input("Please type your answer: ")

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_resume(resume_text):
    prompt = f"""
    You are an AI-powered interview assistant. Based on the following resume, generate 3 relevant interview questions:

    Resume:
    {resume_text}

    Please provide 3 interview questions based on the information in this resume. Return only the questions, one per line.
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=500,
    )

    return response.choices[0].message.content.strip().split("\n")

def detect_cheating(stop_event):
    cap = cv2.VideoCapture(0)
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    movement_threshold = 10000  # Adjust this value based on sensitivity needed
    consecutive_movements = 0
    max_consecutive_movements = 5  # Number of consecutive frames with movement to trigger alert

    while not stop_event.is_set():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        significant_movement = False
        for contour in contours:
            if cv2.contourArea(contour) > movement_threshold:
                significant_movement = True
                break

        if significant_movement:
            consecutive_movements += 1
            if consecutive_movements >= max_consecutive_movements:
                print("Warning: Excessive movement detected. Please focus on the interview.")
                speak("Warning: Excessive movement detected. Please focus on the interview.")
                consecutive_movements = 0
        else:
            consecutive_movements = 0

        frame1 = frame2
        _, frame2 = cap.read()

    cap.release()

def conduct_interview(resume_text):
    questions = ["Hello! How are you?", "Great! Now, could you tell me about yourself?"]
    questions += analyze_resume(resume_text)
    
    stop_event = threading.Event()
    cheating_thread = threading.Thread(target=detect_cheating, args=(stop_event,))
    cheating_thread.start()

    answers = []
    try:
        for i, question in enumerate(questions, 1):
            if question.strip():
                print(f"\nQuestion {i}: {question.strip()}")
                speak(question.strip())
                answer = listen()
                answers.append(answer)
                print(f"Answer recorded: {answer}\n")
    finally:
        stop_event.set()
        cheating_thread.join()

    evaluate_answers(resume_text, questions, answers)

def evaluate_answers(resume_text, questions, answers):
    qa_pairs = [f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))]
    qa_text = "\n".join(qa_pairs)
    
    evaluation_prompt = f"""
    You are an AI-powered interview evaluator. Based on the following resume and the candidate's answers to interview questions, provide an evaluation of the candidate's performance:

    Resume:
    {resume_text}

    Questions and Answers:
    {qa_text}

    Please provide a brief evaluation of the candidate's performance, highlighting strengths and areas for improvement.
    """

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt,
            }
        ],
        model="llama2-70b-4096",
        temperature=0.5,
        max_tokens=500,
    )

    print("\nInterview Evaluation:")
    evaluation = response.choices[0].message.content
    print(evaluation)
    speak("Here's your interview evaluation:")
    speak(evaluation)

if __name__ == "__main__":
    print("Welcome to Pratap Interviewer! World ")
    speak("Welcome to Pratap Interviewer! World")
    
    while True:
        resume_path = input("Please enter the path to your resume PDF file: ")
        if os.path.exists(resume_path) and resume_path.lower().endswith('.pdf'):
            break
        else:
            print("Invalid file path or not a PDF file. Please try again.")
    
    resume_text = extract_text_from_pdf(resume_path)
    conduct_interview(resume_text)