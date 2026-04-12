from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from jobspy import scrape_jobs
from typing import Optional
import pandas as pd
import io, csv, math
import httpx
import os
from datetime import datetime

app = FastAPI(
    title="JobSpy Search Engine API",
    description="Aggregate job listings from LinkedIn, Indeed, Glassdoor, Google, ZipRecruiter & more.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/NaT with None so JSON serialisation never chokes."""
    df = df.where(pd.notnull(df), None)
    for col in ["min_amount", "max_amount", "median_amount"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(int(x)) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None
            )
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x) if not isinstance(x, (type(None), bool, int, float, str)) else x
        )
    return df


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "JobSpy Search Engine API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.get("/api/search", tags=["Jobs"])
def search_jobs(
    search_term: str = Query(..., description="Job title / keyword to search"),
    location: str = Query("", description="City, State or Country"),
    site_name: str = Query(
        "indeed,linkedin,zip_recruiter,google",
        description="Comma-separated: indeed, linkedin, zip_recruiter, glassdoor, google, bayt, naukri",
    ),
    results_wanted: int = Query(20, ge=1, le=100, description="Max results per board"),
    hours_old: int = Query(72, ge=1, le=720, description="Max age of posting in hours"),
    job_type: Optional[str] = Query(None, description="fulltime | parttime | internship | contract"),
    country_indeed: str = Query("USA", description="Country for Indeed (e.g. Canada, UK, Australia)"),
    is_remote: Optional[bool] = Query(None, description="Filter remote-only jobs"),
    distance: Optional[int] = Query(None, description="Distance in miles from location"),
    linkedin_fetch_description: bool = Query(False, description="Fetch full description from LinkedIn (slower)"),
    google_search_term: Optional[str] = Query(None, description="Custom Google Jobs search string"),
):
    try:
        sites = [s.strip() for s in site_name.split(",") if s.strip()]
        kwargs = dict(
            site_name=sites,
            search_term=search_term,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            linkedin_fetch_description=linkedin_fetch_description,
        )
        if location:           kwargs["location"] = location
        if job_type:           kwargs["job_type"] = job_type
        if is_remote is not None: kwargs["is_remote"] = is_remote
        if distance is not None:  kwargs["distance"] = distance
        if google_search_term:    kwargs["google_search_term"] = google_search_term

        df = scrape_jobs(**kwargs)

        if df is None or len(df) == 0:
            return {"jobs": [], "total": 0, "message": "No jobs found - try broader terms or more sites."}

        df = clean_df(df)
        jobs = df.to_dict(orient="records")
        return {"jobs": jobs, "total": len(jobs)}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/export", tags=["Jobs"])
def export_csv(
    search_term: str = Query(...),
    location: str = Query(""),
    site_name: str = Query("indeed,linkedin,zip_recruiter,google"),
    results_wanted: int = Query(50, ge=1, le=100),
    hours_old: int = Query(72),
    job_type: Optional[str] = Query(None),
    country_indeed: str = Query("USA"),
):
    try:
        sites = [s.strip() for s in site_name.split(",") if s.strip()]
        kwargs = dict(
            site_name=sites,
            search_term=search_term,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
        )
        if location:  kwargs["location"] = location
        if job_type:  kwargs["job_type"] = job_type

        df = scrape_jobs(**kwargs)
        if df is None or len(df) == 0:
            raise HTTPException(status_code=404, detail="No jobs found.")

        df = clean_df(df)
        buf = io.StringIO()
        df.to_csv(buf, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
        buf.seek(0)
        filename = f"jobs_{search_term.replace(' ', '_')}.csv"
        return StreamingResponse(
            io.BytesIO(buf.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

class ResumeScoreRequest(BaseModel):
    resume_text: str
    jobs: list[dict]

@app.post("/api/score-resume")
async def score_resume(request: ResumeScoreRequest):
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=400,
            detail="PERPLEXITY_API_KEY not set in environment")
    scored = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for job in request.jobs[:10]:
            prompt = f"""Resume: {request.resume_text[:1200]}
Job: {job.get('title','')} at {job.get('company','')}
Description: {job.get('description','No description')[:600]}
Rate match 0-100. Reply ONLY with valid JSON:
{{"score":75,"reason":"Why good fit","missing":"Skills lacking"}}"""
            try:
                r = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                             "Content-Type": "application/json"},
                    json={"model":"sonar","messages":[
                        {"role":"user","content":prompt}],"max_tokens":150}
                )
                content = r.json()["choices"][0]["message"]["content"]
                import re, json as j
                m = re.search(r'\{.*?\}', content, re.DOTALL)
                if m:
                    d = j.loads(m.group())
                    job["match_score"] = d.get("score", 0)
                    job["match_reason"] = d.get("reason", "")
                    job["match_missing"] = d.get("missing", "")
            except Exception as ex:
                job["match_score"] = 0
            scored.append(job)
    scored.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return {"scored_jobs": scored}

import re

@app.post("/api/score")
async def score_job(data: dict):
    """Score a job against a resume using Perplexity AI."""
    resume      = data.get("resume", "")
    title       = data.get("title", "")
    company     = data.get("company", "")
    location    = data.get("location", "")
    description = data.get("description", "")

    if not resume:
        return {"score": 0, "reasoning": "No resume provided.", "missing": []}
    if not description and not title:
        return {"score": 0, "reasoning": "No job description.", "missing": []}

    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai"
    )

    prompt = f"""You are a professional recruiter. Score this resume against the job posting.
Return ONLY a valid JSON object — no markdown, no text outside the JSON.

RESUME:
{resume[:3000]}

JOB: {title} at {company} ({location})
DESCRIPTION: {description[:2000]}

Return exactly this format:
{{"score": 82, "reasoning": "Strong Python skills match. Missing AWS certification.", "missing": ["AWS", "Kubernetes"]}}

Rules: score 0-100 integer. reasoning 1-2 sentences. missing = up to 5 skill gaps."""

    try:
        response = client.chat.completions.create(
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            import json as _json
            return _json.loads(match.group())
        return {"score": 0, "reasoning": text[:200], "missing": []}
    except Exception as e:
        return {"score": 0, "reasoning": f"Scoring error: {str(e)}", "missing": []}