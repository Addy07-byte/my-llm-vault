# Resume Selection Decision Policy

## Objective
Select the best resume for a given Job Description (JD) to maximize ATS match probability.

## Decision Flow
1. Perform keyword-based matching between JD and existing resumes.
2. Rank resumes using keyword overlap score.
3. If the top resume score ≥ THRESHOLD:
   - Return the existing resume.
4. Else:
   - Fall back to semantic search + LLM generation using user context.
5. If insufficient context exists:
   - Return "Unable to determine a suitable resume".

## Rationale
Keyword matching is prioritized to optimize ATS performance.
Semantic and LLM-based generation is used only as a fallback.

## Notes
- Threshold value is configurable.
- Evaluation metrics (Precision, Recall, MRR) will validate effectiveness.
