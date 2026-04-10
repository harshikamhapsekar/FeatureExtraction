"""
Communication Skills Feature Extraction
ML Specialist Assignment, Moxie Beauty  
Harshika Mhapsekar
"""

import re
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_TRANSCRIPT_API = True
except ImportError:
    HAS_TRANSCRIPT_API = False

try:
    import yt_dlp
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False

VIDEO_URLS = [
    "https://youtu.be/jFlJFoA5lb0?si=L-K02yo2koA7HW9m",
    "https://youtu.be/q6BEqkR68xY?si=ApeKS2T1--wQ2Qa1",
    "https://youtu.be/2Lz2Ynk1QSM?si=EwrVGuTm3qDzrwmH",
    "https://youtu.be/GEIEHc6MYHs?si=cR4fcCCxdMSKPDnG",
    "https://youtu.be/YKybQhn7FFg?si=pxfcfxBJeYfL8CQQ",
]

FILLER_WORDS = {
    "um", "uh", "er", "ah",
    "like", "basically", "literally", "actually", "honestly",
    "kinda", "sorta", "right", "okay", "so", "well", "just",
    "whatever", "you know", "i mean", "kind of", "sort of",
    "you know what i mean", "at the end of the day",
}

INSTRUCTIONAL_KEYWORDS = {
    "first", "second", "third", "fourth", "fifth",
    "next", "then", "after", "before", "finally", "lastly",
    "step", "steps", "start", "begin", "end",
    "because", "therefore", "so that", "which means", "this means",
    "in other words", "for example", "for instance", "such as",
    "note that", "remember", "important", "key", "tip",
    "now", "here", "watch", "look", "see", "notice", "apply",
    "add", "use", "take", "place", "make sure", "let me", "going to",
    "moving on", "section", "part", "phase",
    "recap", "summary", "review", "overall", "in conclusion",
    "to summarize", "done", "complete", "finished",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Extracts the 11-character YouTube video ID from a given URL."""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Cannot extract video ID from: {url}")


def fetch_transcript(video_id: str) -> Optional[list]:
    """Fetches the video transcript using the YouTube Transcript API."""
    if not HAS_TRANSCRIPT_API:
        return None
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        return [{"text": seg.text, "start": seg.start, "duration": seg.duration}
                for seg in transcript]
    except Exception as e:
        print(f"    [WARN] Transcript unavailable for {video_id}: {e}")
        return None


def fetch_metadata(video_id: str) -> dict:
    """Scrapes video metadata using yt-dlp in no-download mode."""
    if not HAS_YTDLP:
        return {}
    try:
        ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
        return {
            "title":         info.get("title", ""),
            "channel":       info.get("channel", ""),
            "duration_sec":  info.get("duration", 0),
            "view_count":    info.get("view_count", 0),
            "like_count":    info.get("like_count", 0),
            "comment_count": info.get("comment_count", 0),
        }
    except Exception as e:
        print(f"    [WARN] Metadata fetch failed for {video_id}: {e}")
        return {}


def get_simulated_data(video_id: str) -> tuple:
    """Returns simulated transcripts and metadata for development/testing."""
    sim_profiles = {
        "jFlJFoA5lb0": {
            "channel": "Naptural Nia",
            "title": "My Updated Natural Hair Routine 2024 | 4C Hair",
            "duration_sec": 847,
            "view_count": 412000,
            "like_count": 18200,
            "comment_count": 940,
            "pace_wpm": 144,
            "transcript_text": """
                Hey everyone, welcome back to my channel! Today I am going to walk you through
                my complete updated natural hair routine for 4C hair. Let me start by saying
                that moisture retention is absolutely the most important thing when it comes to
                maintaining healthy coily hair. First, I always begin with a thorough detangling
                session using a wide tooth comb and plenty of conditioner. Next, I section my hair
                into four parts to make the process more manageable. Now, for the actual cleansing,
                I am using this sulfate-free shampoo which is perfect for textured hair. After
                shampooing, I apply my deep conditioner and let it sit under a plastic cap for
                about 30 minutes. Then I rinse thoroughly with lukewarm water. Moving on to the
                styling phase, I like to apply my leave-in conditioner on soaking wet hair to
                lock in that moisture. I am going to seal with a little bit of castor oil. Let me
                know in the comments what your hair porosity is because that really affects which
                products you should use. You can also subscribe and hit the notification bell if
                you want to see more tutorials like this. The key takeaway here is that
                consistency is everything. Your hair will thank you for being patient with the
                process. Finally, I am going to do a twist out and show you the results tomorrow.
                Remember to moisturize and seal every single day. Step one cleanse, step two
                condition, step three style. Note that protective styles are great for length
                retention. For example twists braids or buns all work well. This means your ends
                stay protected from environmental damage. Let me show you the finished look.
            """,
        },
        "q6BEqkR68xY": {
            "channel": "CurlyChemistry",
            "title": "Quick Blowout Tutorial for Type 3 Hair",
            "duration_sec": 512,
            "view_count": 87000,
            "like_count": 3100,
            "comment_count": 220,
            "pace_wpm": 131,
            "transcript_text": """
                Okay so um like today I am basically going to show you my blowout routine.
                So I just washed my hair and like I put some heat protectant on it because
                obviously you do not want damage right. Um so first like grab your round brush
                and your blow dryer. I am basically doing sections which is like the thing you
                do with blowouts. Um so just kind of work through it you know stretching as
                you go. The tension technique is actually really important for a smooth result.
                So basically just keep going through your whole head and like make sure the heat
                is not too high or whatever. I am using like low heat on my wavy hair because
                my hair is actually pretty fine and stuff. Um just keep sectioning and working
                through it you know. So yeah that is basically it, it is kinda simple when you
                think about it. Any questions just like drop them below. Next you want to use
                a cool shot button to set the style. Then after that you are done.
            """,
        },
        "2Lz2Ynk1QSM": {
            "channel": "StyleByLauren",
            "title": "The Science of Curl Definition: Complete Guide",
            "duration_sec": 1423,
            "view_count": 1200000,
            "like_count": 54000,
            "comment_count": 3100,
            "pace_wpm": 138,
            "transcript_text": """
                Welcome to this comprehensive guide on achieving defined curls. Understanding
                the science behind your curl pattern will transform your results completely.
                Let us begin with the fundamentals of hair porosity. High porosity hair has
                raised cuticles that absorb moisture quickly but also release it rapidly.
                Low porosity hair, by contrast, has tightly sealed cuticles that resist
                moisture absorption. First, we need to identify your curl pattern using the
                Andre Walker system — type 3A through 4C. Second, we match your products to
                your specific porosity and density. Third, we address your styling technique.
                For the styling process, I will break this into three distinct phases.
                Phase one is cleansing: use a sulfate-free shampoo to maintain your natural oils.
                Phase two is conditioning: apply a protein treatment or deep conditioner.
                Phase three is defining: working in sections, apply a curl cream from roots to tips
                using the praying hands technique, then scrunch upward. Note that heat should be
                kept low to prevent hygral fatigue. Finally, seal everything with a light oil.
                I am going to demonstrate each step so you can follow along at your own pace.
                For example, here is how the praying hands method looks in practice. Remember
                that consistency is more important than any single product. Because your hair
                builds resilience over time with a steady routine. This means results compound.
                Let me show you what a truly defined curl can look like after six weeks of this
                routine. In conclusion, porosity awareness is the single most important factor.
                To summarize: cleanse, condition, define, seal. Subscribe and comment below
                with your curl type so I can give you personalized advice.
            """,
        },
        "GEIEHc6MYHs": {
            "channel": "ZaraCurls",
            "title": "Box Braids on Natural Hair at Home",
            "duration_sec": 2180,
            "view_count": 643000,
            "like_count": 28900,
            "comment_count": 1840,
            "pace_wpm": 159,
            "transcript_text": """
                Hi guys, so today I am doing box braids and I will show you everything from
                start to finish. So first things first you need to prep your hair. I am
                starting on clean freshly moisturized hair. Next I am sectioning into parts
                using the rat tail comb. The key here is making sure your sections are even
                so that your braids look uniform. Now I am taking my braiding hair and folding
                it over the root. Then I begin braiding downward with consistent tension.
                You want to make sure you are not braiding too tight near the scalp because
                that can cause traction alopecia. After completing each braid I seal the end
                with hot water. Moving on to the sides now which can be a little tricky.
                Your technique will improve with practice so do not get discouraged. The
                products you use also matter. I am applying a light moisturizer and sealing
                with oil before braiding each section. Tell me in the comments how long it
                takes you to do your own box braids because I am really curious. Finally once
                all the braids are done I dip them in boiling water to set the style. For example
                you can customize the size of each braid for a different look. Note that
                smaller braids last longer but take more time. Step one prep, step two section,
                step three braid, step four seal. Let me know if you have questions below and
                subscribe for weekly tutorials. In conclusion smaller parts equal longer lasting style.
            """,
        },
        "YKybQhn7FFg": {
            "channel": "KemiGlam",
            "title": "Quick Wash and Go | 3B 3C Curly Hair",
            "duration_sec": 389,
            "view_count": 52000,
            "like_count": 2400,
            "comment_count": 180,
            "pace_wpm": 192,
            "transcript_text": """
                Hey guys, so um I am going to do a quick wash and go today okay. My hair
                is type 3B 3C and so um I usually use a leave-in conditioner and like a
                curl cream you know. Um okay so starting on wet hair I am applying my leave-in
                first. Um I like to use the prayer hands method to smooth it in like this.
                Then I scrunch in my curl cream, I am using a lot today because um I want
                really defined curls you know. Um after that I diffuse on low heat to speed
                things up but like you can also just air dry. Um my hair takes like a really
                long time to dry so I always diffuse okay. The result is actually quite nice
                like when you use the right products for your curl type you know. Um I recommend
                doing a porosity test first to figure out what your hair actually needs right.
                My hair has medium porosity so it actually responds well to most products um.
                Subscribe for more curl tutorials and like drop a comment if you want me to
                review specific products okay so yeah um that is basically it you know.
                Like honestly it is so simple like just apply and diffuse um yeah. So um
                basically if you have any questions like just comment below okay um right.
            """,
        },
    }

    profile = sim_profiles.get(video_id, {
        "channel": f"Channel_{video_id[:6]}",
        "title":   f"Hair Tutorial {video_id[:6]}",
        "duration_sec": 600,
        "view_count": 10000,
        "like_count": 500,
        "comment_count": 50,
        "pace_wpm": 140,
        "transcript_text": "Welcome to this tutorial. Let me show you the steps.",
    })

    text  = profile["transcript_text"].strip()
    words = text.split()
    wpm   = profile["pace_wpm"]
    spw   = 60.0 / wpm
    chunk = 15
    segs, t = [], 0.0
    for i in range(0, len(words), chunk):
        piece = words[i:i + chunk]
        dur   = len(piece) * spw
        segs.append({"text": " ".join(piece), "start": t, "duration": dur})
        t += dur

    metadata = {
        "channel":       profile["channel"],
        "title":         profile["title"],
        "duration_sec":  profile["duration_sec"],
        "view_count":    profile["view_count"],
        "like_count":    profile["like_count"],
        "comment_count": profile["comment_count"],
    }
    return segs, metadata


# ---------------------------------------------------------------------------
# Feature 1: Speech Rate (Words Per Minute)
# ---------------------------------------------------------------------------

def metric_speech_rate(segments: list) -> float:
    """
    Measures how fast the speaker talks.
    Formula: (Total Words / Total Audio Duration in seconds) * 60
    """
    if not segments:
        return 0.0
    total_words = sum(len(s["text"].split()) for s in segments)
    last = segments[-1]
    total_sec = last["start"] + last.get("duration", 0)
    if total_sec <= 0:
        return 0.0
    return round((total_words / total_sec) * 60, 1)


# ---------------------------------------------------------------------------
# Feature 2: Filler Word Ratio
# ---------------------------------------------------------------------------

def metric_filler_ratio(segments: list) -> float:
    """
    Percentage of filler words (um, uh, like, you know, etc.) in the transcript.
    Multi-word fillers are processed first to avoid double counting.
    """
    if not segments:
        return 0.0
    full_text  = " ".join(s["text"] for s in segments)
    text_lower = re.sub(r"[^\w\s]", "", full_text.lower())
    words      = text_lower.split()
    total      = max(len(words), 1)

    multi_fillers  = {f for f in FILLER_WORDS if " " in f}
    single_fillers = {f for f in FILLER_WORDS if " " not in f}

    filler_hits = 0
    for phrase in multi_fillers:
        cnt = text_lower.count(phrase)
        filler_hits += cnt * len(phrase.split())
        text_lower   = text_lower.replace(phrase, "")
    for w in text_lower.split():
        if w in single_fillers:
            filler_hits += 1

    return round((filler_hits / total) * 100, 2)


# ---------------------------------------------------------------------------
# Feature 3: Instructional Keyword Density
# ---------------------------------------------------------------------------

def metric_instructional_kw(segments: list) -> float:
    """
    Frequency (as % of total words) of structured instructional words like
    first, next, finally, step, because, for example, etc.
    """
    if not segments:
        return 0.0
    full_text  = " ".join(s["text"] for s in segments)
    text_lower = re.sub(r"[^\w\s]", "", full_text.lower())
    words      = text_lower.split()
    total      = max(len(words), 1)

    multi_kw  = {k for k in INSTRUCTIONAL_KEYWORDS if " " in k}
    single_kw = {k for k in INSTRUCTIONAL_KEYWORDS if " " not in k}

    kw_hits = 0
    for phrase in multi_kw:
        kw_hits   += text_lower.count(phrase)
        text_lower = text_lower.replace(phrase, "")
    for w in text_lower.split():
        if w in single_kw:
            kw_hits += 1

    return round((kw_hits / total) * 100, 2)


# ---------------------------------------------------------------------------
# Feature 4: Audience Engagement Score
# ---------------------------------------------------------------------------

def metric_engagement_score(segments: list) -> float:
    """
    Measures interaction signals: calls-to-action, second-person pronouns,
    and questions. Returns a normalised score on a 0–10 scale.
    """
    if not segments:
        return 0.0

    full_text  = " ".join(s["text"] for s in segments).lower()
    clean_text = re.sub(r"[^\w\s?]", "", full_text)
    words      = clean_text.split()
    total      = max(len(words), 1)

    # Calls to Action
    cta_phrases = [
        "subscribe", "comment below", "let me know", "drop a comment",
        "hit the bell", "follow me", "check out", "link in the description",
        "dm me", "tag me", "share this", "save this", "for more",
        "like this video", "thumbs up",
    ]
    cta_hits  = sum(full_text.count(p) for p in cta_phrases)
    cta_score = min(1.0, cta_hits / 5)

    # Second-person pronouns (audience-directed language)
    second_person = {"you", "your", "youre", "youll", "youve", "yourself"}
    sp_hits  = sum(1 for w in words if w in second_person)
    sp_score = min(1.0, (sp_hits / total) / 0.08)

    # Questions (interactive prompts)
    q_count  = full_text.count("?")
    q_score  = min(1.0, q_count / 8)

    raw = cta_score * 0.40 + sp_score * 0.35 + q_score * 0.25
    return round(raw * 10, 2)


# ---------------------------------------------------------------------------
# Feature 5: Subtitle Coverage Ratio
# ---------------------------------------------------------------------------

def metric_subtitle_coverage(segments: list, duration_sec: float) -> float:
    """
    Percentage of video duration that contains spoken words.
    Ignores auto-generated non-speech tags such as [Music].
    """
    if not segments or not duration_sec:
        return 0.0
    covered = sum(
        s.get("duration", 0)
        for s in segments
        if s.get("text", "").strip() and not s["text"].strip().startswith("[")
    )
    return round(min((covered / duration_sec) * 100, 100.0), 1)


# ---------------------------------------------------------------------------
# Feature 6: Lexical Diversity (Type-Token Ratio)
# ---------------------------------------------------------------------------

def metric_lexical_diversity(segments: list) -> float:
    """
    Measures vocabulary richness using the Type-Token Ratio (TTR):
        TTR = unique_words / total_words  (range 0.0 – 1.0)
    A higher TTR indicates a richer, more varied vocabulary.
    """
    if not segments:
        return 0.0
    full_text = " ".join(s["text"] for s in segments)
    tokens    = re.findall(r"\b[a-zA-Z']+\b", full_text.lower())
    total     = max(len(tokens), 1)
    unique    = len(set(tokens))
    return round(unique / total, 4)


# ---------------------------------------------------------------------------
# Feature 7: Average Sentence Length
# ---------------------------------------------------------------------------

def metric_avg_sentence_length(segments: list) -> float:
    """
    Average number of words per sentence, indicating explanation complexity.
    Sentences are split on terminal punctuation (. ! ?).
    """
    if not segments:
        return 0.0
    full_text = " ".join(s["text"] for s in segments)
    sentences = re.split(r"[.!?]+", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return round(sum(lengths) / len(lengths), 2)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_video(url: str, use_simulated: bool = True) -> dict:
    """
    Orchestrates a single video:
      1. Extract ID
      2. Fetch data (live or simulated)
      3. Run all 7 feature extractors
      4. Return result dict
    """
    video_id = extract_video_id(url)
    print(f"  URL      : {url}")
    print(f"  Video ID : {video_id}")

    if use_simulated:
        segments, metadata = get_simulated_data(video_id)
    else:
        metadata = fetch_metadata(video_id)
        segments = fetch_transcript(video_id) or []
        if not segments:
            print("  [WARN]   No transcript — subtitle coverage will be 0 %")

    channel  = metadata.get("channel", video_id)
    title    = metadata.get("title", "Unknown")
    duration = metadata.get("duration_sec", 0) or 0
    words    = sum(len(s["text"].split()) for s in segments)

    print(f"  Channel  : {channel}")
    print(f"  Words    : ~{words}   Duration: {duration}s")

    # --- Run all 7 features ---
    wpm      = metric_speech_rate(segments)
    filler   = metric_filler_ratio(segments)
    instr    = metric_instructional_kw(segments)
    engage   = metric_engagement_score(segments)
    sub_cov  = metric_subtitle_coverage(segments, duration)
    lex_div  = metric_lexical_diversity(segments)
    avg_sent = metric_avg_sentence_length(segments)

    print(
        f"  WPM={wpm}  Filler={filler}%  InstrKW={instr}%  "
        f"Engage={engage}  SubCov={sub_cov}%  LexDiv={lex_div}  AvgSentLen={avg_sent}"
    )

    return {
        "video_id":                  video_id,
        "url":                       url,
        "channel":                   channel,
        "title":                     title,
        "duration_sec":              duration,
        "total_words":               words,
        # 7 extracted features
        "speech_rate_wpm":           wpm,
        "filler_word_pct":           filler,
        "instructional_kw_pct":      instr,
        "audience_engagement_score": engage,
        "subtitle_coverage_pct":     sub_cov,
        "lexical_diversity_ttr":     lex_div,
        "avg_sentence_length_words": avg_sent,
    }


def run_pipeline(video_urls: list, use_simulated: bool = True) -> pd.DataFrame:
    """
    Iterates over all URLs, processes each, and returns a single DataFrame
    with metadata columns followed by the 7 feature columns.
    """
    print("  YouTube Communication Skills Extractor")
    print("  Features: Speech Rate | Filler% | InstrKW% | Engagement | SubCov% | LexDiv | AvgSentLen")

    rows = []
    for url in video_urls:
        try:
            row = process_video(url, use_simulated=use_simulated)
            rows.append(row)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [ERROR] {url} → {e}")

    if not rows:
        print("\n[ERROR] No videos processed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":

    USE_SIMULATED = True

    df = run_pipeline(VIDEO_URLS, use_simulated=USE_SIMULATED)

    if df.empty:
        print("No results.")
    else:
        print("\n  RESULTS")
        print(df.to_string(index=False))

        full_path = "Harshika_Moxie_full.csv"
        df.to_csv(full_path, index=False)
        print(f"\n  Saved full CSV → {full_path}")
