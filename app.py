"""
Streamlit Web Application for University Major Recommendation System.
"""
import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.recommender import MajorRecommender

# Page Configuration
st.set_page_config(
    page_title="Academic Major Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def load_css(css_filename: str = "style.css"):
    """Loads external CSS stylesheet into Streamlit."""
    css_path = os.path.join(os.path.dirname(__file__), css_filename)
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    return MajorRecommender()


def main():
    load_css()

    # Solid Top Bar Header (Single color #091540, edge-to-edge, 0 margin top)
    st.markdown("""
    <div class="top-bar-header">
        <div class="top-bar-title">
            نظام توصية التخصصات الجامعية | Major Recommender System
        </div>
        <div class="top-bar-subtitle">
            مساعد ذكي قائم على التعلم الآلي لإرشاد طلبة التوجيهي نحو التخصص الجامعي الأمثل وفقًا لمهاراتهم واهتماماتهم
        </div>
    </div>
    """, unsafe_allow_html=True)

    recommender = load_engine()

    # Navigation tabs attached to top bar zone
    tabs = st.tabs(["Recommendation Tool", "Model Analytics & Insights", "System Overview"])

    with tabs[0]:
        col_left, col_right = st.columns([6, 6], gap="medium")

        with col_left:
            st.markdown('<div class="panel-header">Student Profile & Preferences</div>', unsafe_allow_html=True)
            
            in_sub1, in_sub2 = st.columns(2, gap="small")
            
            with in_sub1:
                st.markdown("##### Academic Profile")
                gpa = st.slider(
                    "Tawjihi GPA (معدل التوجيهي)",
                    min_value=60.0,
                    max_value=100.0,
                    value=88.5,
                    step=0.1
                )

                thinking_style = st.selectbox(
                    "Thinking Style",
                    options=["تفكير منطقي", "تفكير تحليلي"],
                    format_func=lambda x: "Logical (منطقي)" if x == "تفكير منطقي" else "Analytical (تحليلي)"
                )

                personality_type = st.selectbox(
                    "Personality Type",
                    options=["اجتماعي (بتحب تختلط بالناس بكثرة )", "منطوي (ما بتحب كثير تختلط بالناس)"],
                    format_func=lambda x: "Extroverted (اجتماعي)" if "اجتماعي" in x else "Introverted (منطوي)"
                )

                preferred_study = st.selectbox(
                    "Study Method",
                    options=["مزيج بين النظري والعملي", "دراسة فيها عملي (تطبيق)", "نظري"],
                    format_func=lambda x: "Hybrid (مزيج)" if "مزيج" in x else ("Practical (عملي)" if "عملي" in x else "Theoretical (نظري)")
                )

                preferred_work = st.selectbox(
                    "Work Style",
                    options=["الاثنين معا", "العمل الجماعي (ضمن فريق)", "العمل الفردي"],
                    format_func=lambda x: "Teamwork (جماعي)" if "الجماعي" in x else ("Solo (فردي)" if "الفردي" in x else "Both (معا)")
                )

            with in_sub2:
                st.markdown("##### Field Ratings (1-5)")
                tech_interest = st.slider("Tech (تكنولوجيا)", 1, 5, 4)
                health_interest = st.slider("Health (صحة وطب)", 1, 5, 2)
                business_interest = st.slider("Business (إدارة)", 1, 5, 3)
                arts_interest = st.slider("Arts (فنون)", 1, 5, 3)
                
                top_k = st.number_input("Top K Majors", min_value=3, max_value=10, value=5)

            submit_button = st.button("Generate Recommendations | استخراج التوصيات", type="primary", use_container_width=True)

            if submit_button:
                st.session_state['has_submitted'] = True

        student_input = {
            'GPA': gpa,
            'Interest in Technology': tech_interest,
            'Interest in Health and Medicine': health_interest,
            'Interest in Business and Management': business_interest,
            'Interest in Arts and Creativity': arts_interest,
            'Thinking Style': thinking_style,
            'Personality Type': personality_type,
            'Preferred Study Method': preferred_study,
            'Preferred Work Style': preferred_work
        }

        with col_right:
            if not st.session_state.get('has_submitted', False):
                st.markdown("""
                <div class="placeholder-card">
                    <div class="placeholder-badge">Recommendation Engine</div>
                    <div class="placeholder-title">Recommendations Will Appear Here</div>
                    <div class="placeholder-desc">
                        Adjust the student profile and field ratings on the left, then click <b>Generate Recommendations</b>.
                    </div>
                    <div>
                        <span class="feature-pill">Soft Voting Ensemble</span>
                        <span class="feature-pill">Match Confidence</span>
                        <span class="feature-pill">Domain Insights</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="panel-header">Recommendations Output</div>', unsafe_allow_html=True)

                recommendations = recommender.recommend(student_input, top_k=top_k)

                res_tab1, res_tab2 = st.tabs(["Major Recommendations", "Category & Profile Breakdown"])
                
                with res_tab1:
                    for rec in recommendations:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="rank-badge">Rank #{rec['rank']}</span>
                                <h4 style="margin: 0; color: #091540; font-size: 1rem;">{rec['major']}</h4>
                                <span class="category-tag">{rec['category']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.2rem;">
                                <span style="color: #475569; font-size: 0.82rem;">Match Score: <b style="color: #1B2CC1;">{rec['confidence_pct']}</b></span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(rec['confidence']))

                with res_tab2:
                    df_recs = pd.DataFrame(recommendations)
                    cat_counts = df_recs['category'].value_counts()
                    st.bar_chart(cat_counts, color="#1B2CC1", height=180)

                    interests_data = {
                        'Technology': tech_interest,
                        'Health & Med': health_interest,
                        'Business': business_interest,
                        'Arts & Design': arts_interest
                    }
                    st.dataframe(pd.DataFrame(list(interests_data.items()), columns=['Field', 'Rating (1-5)']), use_container_width=True, height=150)

    with tabs[1]:
        st.subheader("Model Performance & Feature Importances")
        metadata = recommender.metadata

        col_m1, col_m2 = st.columns(2, gap="large")

        with col_m1:
            st.markdown("#### Benchmark Accuracy Comparison")
            metrics = metadata.get('benchmark_metrics', {})
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
            st.dataframe(df_metrics.style.highlight_max(axis=0, color='#ABD2FA'), use_container_width=True)

        with col_m2:
            st.markdown("#### Soft Voting Ensemble Highlights")
            st.markdown("""
            <div class="custom-metric">
                <h5 style="color: #091540; margin-bottom: 0.5rem;">Accuracy Metrics Breakdown</h5>
                <ul style="color: #334155; margin-bottom: 0; padding-left: 1.2rem;">
                    <li><b>Top-3 Accuracy</b>: 87.63% (Correct major in top 3 choices)</li>
                    <li><b>Top-5 Accuracy</b>: 91.80% (Correct major in top 5 choices)</li>
                    <li><b>Top-10 Accuracy</b>: 96.14% (Correct major in top 10 choices)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Feature Importance Breakdown")
        feat_imp = pd.DataFrame(metadata.get('feature_importances', []))
        if not feat_imp.empty:
            st.bar_chart(feat_imp.set_index('Feature'), color="#7692FF")

    with tabs[2]:
        st.subheader("About the Major Recommendation System")
        st.markdown("""
        <div class="custom-metric" style="border-left-color: #1B2CC1;">
            <h4 style="color: #091540;">System Architecture & ML Pipeline</h4>
            <p style="color: #334155; font-size: 0.95rem; line-height: 1.5;">
            This system translates high school graduate profiles (GPA, personal thinking style, work preferences, and ratings across academic areas) into ranked university major recommendations.
            </p>
            <ul style="color: #334155; line-height: 1.5;">
                <li><b>Preprocessing Engine</b>: Cleaned Tawjihi GPA, normalized Arabic survey inputs, and mapped raw survey strings into 61 canonical university majors.</li>
                <li><b>Oversampling</b>: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance rare major distributions.</li>
                <li><b>Machine Learning Ensemble</b>: Soft Voting Classifier combining <b>Extra Trees</b>, <b>Random Forest</b>, <b>XGBoost</b>, and <b>LightGBM</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
