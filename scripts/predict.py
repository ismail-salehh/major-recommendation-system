"""
Command line interface script to run major recommendations for sample or custom student profiles.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommender import MajorRecommender


def main():
    parser = argparse.ArgumentParser(description="University Major Recommendation CLI")
    parser.add_argument("--gpa", type=float, default=90.0, help="Tawjihi GPA (60-100)")
    parser.add_argument("--tech", type=int, default=4, help="Interest in Tech (1-5)")
    parser.add_argument("--health", type=int, default=1, help="Interest in Health (1-5)")
    parser.add_argument("--business", type=int, default=3, help="Interest in Business (1-5)")
    parser.add_argument("--arts", type=int, default=5, help="Interest in Arts (1-5)")
    parser.add_argument("--thinking", type=str, default="تفكير منطقي", choices=["تفكير منطقي", "تفكير تحليلي"])
    parser.add_argument("--personality", type=str, default="منطوي (ما بتحب كثير تختلط بالناس)",
                        choices=["اجتماعي (بتحب تختلط بالناس بكثرة )", "منطوي (ما بتحب كثير تختلط بالناس)"])
    parser.add_argument("--study", type=str, default="دراسة فيها عملي (تطبيق)",
                        choices=["دراسة فيها عملي (تطبيق)", "مزيج بين النظري والعملي", "نظري"])
    parser.add_argument("--work", type=str, default="العمل الفردي",
                        choices=["الاثنين معا", "العمل الجماعي (ضمن فريق)", "العمل الفردي"])
    parser.add_argument("--top_k", type=int, default=5, help="Number of recommendations to return")

    args = parser.parse_args()

    recommender = MajorRecommender()

    student_profile = {
        'GPA': args.gpa,
        'Interest in Technology': args.tech,
        'Interest in Health and Medicine': args.health,
        'Interest in Business and Management': args.business,
        'Interest in Arts and Creativity': args.arts,
        'Thinking Style': args.thinking,
        'Personality Type': args.personality,
        'Preferred Study Method': args.study,
        'Preferred Work Style': args.work
    }

    print("\n" + "=" * 60)
    print("  STUDENT INPUT PROFILE")
    print("=" * 60)
    for k, v in student_profile.items():
        print(f"  • {k:<38}: {v}")

    recs = recommender.recommend(student_profile, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"  TOP {args.top_k} RECOMMENDED MAJORS")
    print("=" * 60)
    for r in recs:
        print(f"  {r['rank']}. {r['major']:<32} | {r['category']:<25} | Match: {r['confidence_pct']}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
