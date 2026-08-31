"""
Gradio Web Interface for University Major Recommendation System.
"""
import os
import sys
import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.recommender import MajorRecommender

recommender = MajorRecommender()


def predict_majors(gpa, tech, health, business, arts, thinking, personality, study, work, top_k):
    student_profile = {
        'GPA': float(gpa),
        'Interest in Technology': int(tech),
        'Interest in Health and Medicine': int(health),
        'Interest in Business and Management': int(business),
        'Interest in Arts and Creativity': int(arts),
        'Thinking Style': thinking,
        'Personality Type': personality,
        'Preferred Study Method': study,
        'Preferred Work Style': work
    }

    recommendations = recommender.recommend(student_profile, top_k=int(top_k))

    output_html = "<div style='font-family: Arial, sans-serif; padding: 10px;'>"
    output_html += "<h3>Top Recommended University Majors</h3>"

    for r in recommendations:
        pct = r['confidence_pct']
        output_html += f"""
        <div style='background: #f8fafc; border-left: 4px solid #4338ca; padding: 12px; margin-bottom: 10px; border-radius: 6px;'>
            <div style='display: flex; justify-content: space-between;'>
                <strong style='font-size: 1.1em; color: #1e1b4b;'>#{r['rank']} - {r['major']}</strong>
                <span style='background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;'>{r['category']}</span>
            </div>
            <p style='margin: 5px 0 0 0; color: #475569;'>Match Confidence: <strong>{pct}</strong></p>
        </div>
        """
    output_html += "</div>"

    labels_dict = {f"{r['major']} ({r['category']})": r['confidence'] for r in recommendations}

    return output_html, labels_dict


with gr.Blocks(title="University Major Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # University Major Recommendation System
    ### نظام توصية التخصصات الجامعية القائم على التعلم الآلي
    Enter the student's high school GPA, personality traits, and field interests to generate personalized university major predictions.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Academic & Personal Characteristics")
            gpa_in = gr.Slider(60.0, 100.0, value=90.0, step=0.1, label="Tawjihi GPA (معدل التوجيهي)")
            thinking_in = gr.Radio(["تفكير منطقي", "تفكير تحليلي"], value="تفكير منطقي", label="Thinking Style (أسلوب التفكير)")
            personality_in = gr.Radio(["اجتماعي (بتحب تختلط بالناس بكثرة )", "منطوي (ما بتحب كثير تختلط بالناس)"],
                                      value="منطوي (ما بتحب كثير تختلط بالناس)", label="Personality Type (طبيعة الشخصية)")
            study_in = gr.Dropdown(["مزيج بين النظري والعملي", "دراسة فيها عملي (تطبيق)", "نظري"],
                                   value="مزيج بين النظري والعملي", label="Preferred Study Method (أسلوب الدراسة)")
            work_in = gr.Dropdown(["الاثنين معا", "العمل الجماعي (ضمن فريق)", "العمل الفردي"],
                                  value="الاثنين معا", label="Preferred Work Style (أسلوب العمل)")

        with gr.Column():
            gr.Markdown("### 2. Area Interest Levels (1 = Low, 5 = High)")
            tech_in = gr.Slider(1, 5, value=4, step=1, label="Technology Interest (التكنولوجيا)")
            health_in = gr.Slider(1, 5, value=1, step=1, label="Health & Medical Interest (المجال الصحي والطبي)")
            business_in = gr.Slider(1, 5, value=3, step=1, label="Business Interest (مجال الأعمال والإدارة)")
            arts_in = gr.Slider(1, 5, value=3, step=1, label="Arts & Creativity Interest (المجالات الفنية والإبداعية)")
            top_k_in = gr.Slider(3, 10, value=5, step=1, label="Top Recommendations Count")

            btn = gr.Button("Predict Recommendations", variant="primary")

    with gr.Row():
        out_html = gr.HTML(label="Recommendation Details")
        out_chart = gr.Label(label="Confidence Distribution", num_top_classes=5)

    btn.click(
        fn=predict_majors,
        inputs=[gpa_in, tech_in, health_in, business_in, arts_in, thinking_in, personality_in, study_in, work_in, top_k_in],
        outputs=[out_html, out_chart]
    )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860)
