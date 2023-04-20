import os

from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_cors import CORS
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings
import pandas as pd

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        question = request.form["question"]
        answer = get_answer(question)
        return redirect(url_for("index", result=answer))

    result = request.args.get("result")
    return render_template("index.html", result=result)

@app.route("/ask")
def ask():
    question = request.args.get("question")
    answer = get_answer(question)
    return jsonify({ 'question': question, 'answer': answer })

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


def get_answer(question):
    answer = answer_question(
        df, question=question, debug=False, max_len=1800, max_tokens=300,
    )
    return answer


def answer_question(
    df,
    *,
    question,
    debug,
    max_len,
    max_tokens,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        prompt = ("Answer the question based on the context below, and if the question can't be "
                  "answered based on the context, say \"I don't know\". Please also include "
                  "URLs to the original articles by simply appending (Source: <url>) at the end of "
                  "your answer.\n\n"
                  f"Context: {context}\n\n---\n\n"
                  f"Question: {question}\nAnswer:")

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            temperature=0,
            stop=stop_sequence,
            max_tokens=max_tokens,
        )
        print(f'{response=}')
        total_tokens = response['usage']['total_tokens']
        print(f"{total_tokens=}, estimated_cost=${total_tokens/1000*0.002:.3f}")
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""


def create_context(
    question, df, max_len=1800,  # size='ada',
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(
        input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(
        q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

        
    # Return the context
    return "\n\n###\n\n".join(returns)

if __name__ == "__main__":
    app.run(host="localhost", port=4000, debug=True)
