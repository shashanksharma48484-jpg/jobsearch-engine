from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from jobspy import scrape_jobs
from typing import Optional
import pandas as pd
import io, csv, math, re, json, asyncio, os
from concurrent.futures import ThreadPoolExecutor
import httpx
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(
    title="JobSearch Pro API",
    description="Job listings + AI Smart Search + Resume Scoring",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# ── Helpers ──────────────────────────────────────────────────────────────────

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

def df_to_jobs(df: pd.DataFrame) -> list:
    """Convert cleaned DataFrame to list of job dicts (used by smart-search)."""
    df = clean_df(df)
    jobs = []
    for _, row in df.iterrows():
        desc = str(row.get("description", "") or "")
        jobs.append({
            "title":         str(row.get("title", "") or ""),
            "company":       str(row.get("company", "") or ""),
            "location":      str(row.get("location", "") or ""),
            "job_type":      str(row.get("job_type", "") or ""),
            "is_remote":     bool(row.get("is_remote", False)),
            "min_amount":    row.get("min_amount"),
            "max_amount":    row.get("max_amount"),
            "median_amount": row.get("median_amount"),
            "currency":      str(row.get("currency", "") or ""),
            "interval":      str(row.get("interval", "") or ""),
            "site":          str(row.get("site", "") or ""),
            "date_posted":   str(row.get("date_posted", "") or ""),
            "job_url":       str(row.get("job_url", "") or ""),
            "description":   desc[:3000],
        })
    return jobs

# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "JobSearch Pro API v3 is running"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

# ── Search ───────────────────────────────────────────────────────────────────

@app.get("/api/search", tags=["Jobs"])
def search_jobs(
    search_term: str = Query(..., description="Job title / keyword to search"),
    location: str = Query("", description="City, State or Country"),
    site_name: str = Query(
        "indeed,linkedin,zip_recruiter,google",
        description="Comma-separated: indeed, linkedin, zip_recruiter, glassdoor, google, bayt",
    ),
    results_wanted: int = Query(20, ge=1, le=100),
    hours_old: int = Query(72, ge=1, le=720),
    job_type: Optional[str] = Query(None, description="fulltime | parttime | internship | contract"),
    country_indeed: str = Query("USA"),
    is_remote: Optional[bool] = Query(None),
    distance: Optional[int] = Query(None),
    linkedin_fetch_description: bool = Query(False),
    google_search_term: Optional[str] = Query(None),
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
        if location:              kwargs["location"] = location
        if job_type:              kwargs["job_type"] = job_type
        if is_remote is not None: kwargs["is_remote"] = is_remote
        if distance is not None:  kwargs["distance"] = distance
        if google_search_term:    kwargs["google_search_term"] = google_search_term

        df = scrape_jobs(**kwargs)
        if df is None or len(df) == 0:
            return {"jobs": [], "total": 0, "message": "No jobs found — try broader terms."}

        df = clean_df(df)
        return {"jobs": df.to_dict(orient="records"), "total": len(df)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ── Export CSV ───────────────────────────────────────────────────────────────

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
        kwargs = dict(site_name=sites, search_term=search_term,
                      results_wanted=results_wanted, hours_old=hours_old,
                      country_indeed=country_indeed)
        if location: kwargs["location"] = location
        if job_type: kwargs["job_type"] = job_type

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

# ── Score Resume — bulk, up to 10 jobs ───────────────────────────────────────

class ResumeScoreRequest(BaseModel):
    resume_text: str
    jobs: list[dict]

@app.post("/api/score-resume", tags=["AI Scoring"])
async def score_resume(request: ResumeScoreRequest):
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=400, detail="PERPLEXITY_API_KEY not set")
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
                    json={"model": "sonar",
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 150}
                )
                content = r.json()["choices"][0]["message"]["content"]
                m = re.search(r'\{.*?\}', content, re.DOTALL)
                if m:
                    d = json.loads(m.group())
                    job["match_score"]   = d.get("score", 0)
                    job["match_reason"]  = d.get("reason", "")
                    job["match_missing"] = d.get("missing", "")
            except Exception:
                job["match_score"] = 0
            scored.append(job)
    scored.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return {"scored_jobs": scored}

# ── Score — single job ────────────────────────────────────────────────────────

@app.post("/api/score", tags=["AI Scoring"])
async def score_job(data: dict):
    resume      = data.get("resume", "")
    title       = data.get("title", "")
    company     = data.get("company", "")
    location    = data.get("location", "")
    description = data.get("description", "")

    if not resume:
        return {"score": 0, "reasoning": "No resume provided.", "missing": []}

    job_ctx  = f"Title: {title}\nCompany: {company}\nLocation: {location}\n"
    job_ctx += f"Description: {description[:2000]}" if description else "(No JD — title only)"

    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
    prompt = f"""Expert ATS recruiter. Score resume vs job. Return ONLY valid JSON.
RESUME: {resume[:3000]}
JOB: {job_ctx}
Return: {{"score":82,"reasoning":"Strong PM background. Missing PMP.","missing":["PMP"]}}
Rules: score=0-100 integer. reasoning=1-2 sentences. missing=up to 5 gaps."""

    try:
        r     = client.chat.completions.create(model="sonar",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, temperature=0.2)
        text  = r.choices[0].message.content.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        return json.loads(match.group()) if match else {"score": 0, "reasoning": text[:200], "missing": []}
    except Exception as e:
        return {"score": 0, "reasoning": f"Scoring error: {e}", "missing": []}

# ── Smart Search — AI generates 6 searches from resume ───────────────────────

@app.post("/api/smart-search", tags=["AI Scoring"])
async def smart_search(data: dict):
    resume           = data.get("resume", "")
    location         = data.get("location", "Toronto, Canada")
    country          = data.get("country", "Canada")
    results_per_term = data.get("results_per_search", 15)
    hours_old        = data.get("hours_old", 168)

    if not resume:
        return {"status": "error", "message": "Resume required.", "jobs": [], "searches": []}

    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
    ep = f"""Analyze this resume. Generate exactly 6 targeted job search queries. Return ONLY valid JSON.
RESUME: {resume[:3000]}
Return: {{"searches":[
  {{"term":"IT Project Coordinator Banking","reason":"Primary title + industry"}},
  {{"term":"Business Process Automation Analyst","reason":"Core skill match"}},
  {{"term":"PMP Project Manager Financial Services","reason":"Cert + sector"}},
  {{"term":"Operations Analyst Fintech","reason":"Adjacent title"}},
  {{"term":"Google Apps Script Developer","reason":"Technical specialty"}},
  {{"term":"Senior Analyst Process Improvement","reason":"Growth target"}}
]}}
Rules: 2-5 words per term. Cover: titles, skills, certs, industries, seniority."""

    try:
        r       = client.chat.completions.create(model="sonar",
                    messages=[{"role": "user", "content": ep}],
                    max_tokens=600, temperature=0.3)
        text    = r.choices[0].message.content.strip()
        m       = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return {"status": "error", "message": "Could not parse resume.", "jobs": [], "searches": []}
        searches = json.loads(m.group()).get("searches", [])
    except Exception as e:
        return {"status": "error", "message": f"AI error: {e}", "jobs": [], "searches": []}

    async def one(term, reason):
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                df = await loop.run_in_executor(pool, lambda: scrape_jobs(
                    site_name=["indeed", "zip_recruiter", "glassdoor"],
                    search_term=term, location=location,
                    country_indeed=country,
                    results_wanted=results_per_term,
                    hours_old=hours_old, distance=50))
            jobs = df_to_jobs(df)
            for j in jobs:
                j["search_term"]   = term
                j["search_reason"] = reason
            return jobs
        except:
            return []

    batches = await asyncio.gather(*[one(s["term"], s.get("reason", "")) for s in searches])
    seen, out = set(), []
    for batch in batches:
        for job in batch:
            url = job.get("job_url", "")
            if url and url not in seen:
                seen.add(url); out.append(job)
            elif not url:
                out.append(job)

    return {"status": "ok", "count": len(out), "searches": searches, "jobs": out}

# ── Search (POST) — accepts any JSON body, flexible field names ───────────────

@app.post("/api/search", tags=["Jobs"])
def search_jobs_post(data: dict):
    try:
        raw_sites = data.get("site_name", "indeed,linkedin,zip_recruiter,google")
        if isinstance(raw_sites, list):
            sites = raw_sites
        else:
            sites = [s.strip() for s in str(raw_sites).split(",") if s.strip()]

        kwargs = dict(
            site_name=sites,
            search_term=str(data.get("search_term", "")),
            results_wanted=int(data.get("results_wanted", 20)),
            hours_old=int(data.get("hours_old", 72)),
            country_indeed=str(data.get("country_indeed", "Canada")),
            linkedin_fetch_description=bool(data.get("linkedin_fetch_description", False)),
        )

        location = data.get("location", "")
        if location: kwargs["location"] = str(location)

        job_type = data.get("job_type")
        if job_type: kwargs["job_type"] = str(job_type)

        is_remote = data.get("is_remote")
        if is_remote is not None: kwargs["is_remote"] = bool(is_remote)

        distance = data.get("distance")
        if distance is not None: kwargs["distance"] = int(distance)

        google_search_term = data.get("google_search_term")
        if google_search_term: kwargs["google_search_term"] = str(google_search_term)

        if not kwargs["search_term"]:
            return {"jobs": [], "total": 0, "message": "search_term is required."}

        df = scrape_jobs(**kwargs)
        if df is None or len(df) == 0:
            return {"jobs": [], "total": 0, "message": "No jobs found — try broader terms."}

        df = clean_df(df)
        return {"jobs": df.to_dict(orient="records"), "total": len(df)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))