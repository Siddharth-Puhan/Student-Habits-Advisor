from flask import Flask, render_template, request
import joblib
import numpy as np 
import pandas as pd

app = Flask(__name__)

# Load saved components
model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
df = joblib.load('dataframe.pkl')  # if you're using it for means
df_cluster = pd.read_csv('cluster_means.csv')
cluster_means = pd.read_csv('cluster_means.csv', index_col=0)

# Cluster labels
cluster_labels = {
    0: "Stressed Minimalists",
    1: "Disengaged Energizers",
    2: "Well-Rounded Achievers"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and convert input
        study_hours = float(request.form['study_hours'])
        mental_health = float(request.form['mental_health'])
        exercise = float(request.form['exercise'])
        attendance = float(request.form['attendance'])
        sleep = float(request.form['sleep'])
        social_media = float(request.form['social_media'])
        netflix = float(request.form['netflix'])
    except ValueError:
        return render_template('result.html', cluster="Invalid Input", suggestions=["Please enter valid numbers."])

    # Prepare input
    input_data = [[
        study_hours,
        mental_health,
        exercise,
        attendance,
        sleep,
        social_media,
        netflix
    ]]

    # Scale and predict
    scaled_input = scaler.transform(input_data)
    cluster = model.predict(scaled_input)[0]
    cluster_name = cluster_labels.get(cluster, "Unknown Cluster")

    # Suggestions using cluster-specific or full dataset means
    suggestions = []

    if study_hours < cluster_means.loc[cluster, 'study_hours_per_day'] :
        suggestions.append("📚 Consider increasing your study time.")
    if mental_health < cluster_means.loc[cluster, 'mental_health_rating']:
        suggestions.append("🧠 Focus on improving your mental well-being.")
    if social_media > cluster_means.loc[cluster, 'social_media_hours']:
        suggestions.append("📵 Try to reduce time spent on social media.")
    if exercise < cluster_means.loc[cluster, 'exercise_frequency']:
        suggestions.append("💪 Increase physical activity for better focus.")
    if attendance < cluster_means.loc[cluster, 'attendance_percentage']:
        suggestions.append("📅 Try to attend more classes regularly.")
    if sleep < cluster_means.loc[cluster, 'sleep_hours']:
        suggestions.append("🛌 Improve your sleep routine for better recovery.")
    if netflix > cluster_means.loc[cluster, 'netflix_hours']:
        suggestions.append("🎯 Limit your entertainment time during study days.")

    # Pass values for graph
    feature_names = list(cluster_means.columns)
    user_values = input_data[0]
    cluster_avg = cluster_means.loc[int(cluster)].values.tolist()

    return render_template('result.html',
                           cluster=cluster_name,
                           suggestions=suggestions,
                           features=feature_names,
                           user_values=user_values,
                           cluster_avg=cluster_avg)

if __name__ == '__main__':
    app.run(debug=True)