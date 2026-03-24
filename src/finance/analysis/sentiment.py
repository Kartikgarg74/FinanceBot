"""News & sentiment engine — fetches news, scores sentiment, maps to assets."""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    title: str
    body: str = ""
    source: str = ""
    url: str = ""
    published_at: datetime = field(default_factory=datetime.utcnow)
    tickers: list[str] = field(default_factory=list)
    # AI-scored fields (filled after scoring)
    sentiment_score: float = 0.0  # -5 to +5
    confidence: float = 0.0  # 0-100
    category: str = ""  # company_specific, sector, macro, etc.
    timeframe: str = ""  # immediate, short_term, long_term


@dataclass
class TickerSentiment:
    ticker: str
    score: float  # Weighted aggregated score (-5 to +5)
    confidence: float
    article_count: int
    latest_headline: str = ""


class NewsFetcher:
    """Fetches financial news from free APIs."""

    def __init__(self, config: dict):
        self.config = config.get("news", {})
        self.sources = self.config.get("sources", {})
        self._client = httpx.Client(timeout=15.0)

    def fetch_finnhub_news(self, category: str = "general") -> list[NewsArticle]:
        """Fetch news from Finnhub (60 calls/min free)."""
        api_key = self.sources.get("finnhub", {}).get("api_key", "")
        if not api_key:
            return []

        articles = []
        try:
            resp = self._client.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": api_key},
            )
            resp.raise_for_status()
            for item in resp.json()[:20]:
                articles.append(NewsArticle(
                    title=item.get("headline", ""),
                    body=item.get("summary", ""),
                    source="finnhub",
                    url=item.get("url", ""),
                    published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                    tickers=item.get("related", "").split(",") if item.get("related") else [],
                ))
            logger.info("Finnhub: fetched %d articles (%s)", len(articles), category)
        except Exception as e:
            logger.error("Finnhub fetch failed: %s", e)
        return articles

    def fetch_finnhub_company_news(self, ticker: str, days: int = 3) -> list[NewsArticle]:
        """Fetch company-specific news from Finnhub."""
        api_key = self.sources.get("finnhub", {}).get("api_key", "")
        if not api_key:
            return []

        articles = []
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            resp = self._client.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": ticker,
                    "from": start.strftime("%Y-%m-%d"),
                    "to": end.strftime("%Y-%m-%d"),
                    "token": api_key,
                },
            )
            resp.raise_for_status()
            for item in resp.json()[:10]:
                articles.append(NewsArticle(
                    title=item.get("headline", ""),
                    body=item.get("summary", ""),
                    source="finnhub",
                    url=item.get("url", ""),
                    published_at=datetime.fromtimestamp(item.get("datetime", 0)),
                    tickers=[ticker],
                ))
            logger.info("Finnhub company news: %d articles for %s", len(articles), ticker)
        except Exception as e:
            logger.error("Finnhub company news failed for %s: %s", ticker, e)
        return articles

    def fetch_alpha_vantage_news(self, tickers: list[str] | None = None) -> list[NewsArticle]:
        """Fetch news from Alpha Vantage (25 calls/day free). Returns pre-scored sentiment."""
        api_key = self.sources.get("alpha_vantage", {}).get("api_key", "")
        if not api_key:
            return []

        articles = []
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": api_key,
                "limit": 20,
            }
            if tickers:
                params["tickers"] = ",".join(tickers[:5])

            resp = self._client.get("https://www.alphavantage.co/query", params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("feed", [])[:20]:
                # Alpha Vantage provides per-ticker sentiment!
                article_tickers = []
                best_score = 0
                for ts in item.get("ticker_sentiment", []):
                    article_tickers.append(ts["ticker"])
                    score = float(ts.get("ticker_sentiment_score", 0))
                    if abs(score) > abs(best_score):
                        best_score = score

                published = datetime.strptime(
                    item.get("time_published", "20240101T000000"),
                    "%Y%m%dT%H%M%S"
                )
                overall = float(item.get("overall_sentiment_score", 0))

                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    body=item.get("summary", ""),
                    source="alpha_vantage",
                    url=item.get("url", ""),
                    published_at=published,
                    tickers=article_tickers,
                    # Alpha Vantage gives scores from -1 to 1, scale to -5 to +5
                    sentiment_score=overall * 5,
                    confidence=70,  # Pre-scored, moderate confidence
                ))
            logger.info("Alpha Vantage: fetched %d articles", len(articles))
        except Exception as e:
            logger.error("Alpha Vantage fetch failed: %s", e)
        return articles

    def fetch_reddit_posts(self, subreddit: str = "IndianStreetBets", limit: int = 20) -> list[NewsArticle]:
        """Fetch Reddit posts via PRAW."""
        reddit_config = self.sources.get("reddit", {})
        client_id = reddit_config.get("client_id", "")
        client_secret = reddit_config.get("client_secret", "")
        if not client_id or not client_secret:
            return []

        articles = []
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=reddit_config.get("user_agent", "KartikAI/1.0"),
            )

            sub = reddit.subreddit(subreddit)
            for post in sub.hot(limit=limit):
                articles.append(NewsArticle(
                    title=post.title,
                    body=post.selftext[:500] if post.selftext else "",
                    source="reddit",
                    url=f"https://reddit.com{post.permalink}",
                    published_at=datetime.fromtimestamp(post.created_utc),
                    tickers=[],  # Will be extracted by NER/LLM
                ))
            logger.info("Reddit r/%s: fetched %d posts", subreddit, len(articles))
        except Exception as e:
            logger.error("Reddit fetch failed for r/%s: %s", subreddit, e)
        return articles

    def close(self):
        self._client.close()


class SentimentScorer:
    """Scores news sentiment using AI and aggregates per ticker."""

    def __init__(self, ai_router, news_config: dict):
        self.ai = ai_router
        self.config = news_config
        self.macro_rules = news_config.get("macro_impact_rules", {})
        self.sector_tickers = news_config.get("sector_tickers", {})
        self.source_weights = news_config.get("source_weights", {})

    def score_article(self, article: NewsArticle) -> NewsArticle:
        """Score a single article using Claude Haiku. Updates article in-place."""
        if article.sentiment_score != 0 and article.source == "alpha_vantage":
            return article  # Already scored by Alpha Vantage

        from src.ai.prompts.sentiment import build_sentiment_prompt, SENTIMENT_SYSTEM_PROMPT
        prompt = build_sentiment_prompt(article.title, article.body, article.tickers)

        try:
            result = self.ai.route_json(
                task="news_sentiment",
                prompt=prompt,
                system_prompt=SENTIMENT_SYSTEM_PROMPT,
                max_tokens=512,
            )
            from src.utils.security import validate_llm_score, validate_llm_confidence
            article.sentiment_score = validate_llm_score(result.get("overall_score", 0), -5, 5, 0)
            article.confidence = validate_llm_confidence(result.get("confidence", 50))
            article.category = str(result.get("category", ""))[:50]
            article.timeframe = str(result.get("timeframe", "short_term"))[:20]

            # Extract tickers mentioned by AI
            for affected in result.get("affected_tickers", []):
                t = affected.get("ticker", "")
                if t and t not in article.tickers:
                    article.tickers.append(t)

        except Exception as e:
            logger.error("Sentiment scoring failed for '%s': %s", article.title[:50], e)

        return article

    def score_batch(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Score a batch of articles."""
        scored = []
        for article in articles:
            scored.append(self.score_article(article))
            time.sleep(0.1)  # Gentle rate limiting
        return scored

    def check_macro_triggers(self, article: NewsArticle) -> dict[str, float]:
        """Check if article matches any macro impact rules. Returns sector->score mapping."""
        text = f"{article.title} {article.body}".lower()
        impacts = {}

        for rule_name, rule in self.macro_rules.items():
            triggers = rule.get("triggers", [])
            for trigger in triggers:
                if trigger.lower() in text:
                    for sector, score in rule.get("impact", {}).items():
                        impacts[sector] = impacts.get(sector, 0) + score
                    break  # Don't double-count same rule

        return impacts

    def expand_sector_to_tickers(self, sector_impacts: dict[str, float]) -> dict[str, float]:
        """Convert sector impacts to individual ticker impacts."""
        ticker_impacts = {}
        for sector, score in sector_impacts.items():
            tickers = self.sector_tickers.get(sector, [])
            for ticker in tickers:
                ticker_impacts[ticker] = ticker_impacts.get(ticker, 0) + score
        return ticker_impacts

    def aggregate_sentiment(
        self,
        articles: list[NewsArticle],
        asset_type: str = "stock",
    ) -> dict[str, TickerSentiment]:
        """
        Aggregate sentiment across all articles per ticker.
        Uses exponential decay based on recency + source credibility.
        """
        now = datetime.utcnow()
        half_life_hours = 6 if asset_type == "crypto" else 24
        decay_lambda = 0.693 / half_life_hours

        # Collect weighted scores per ticker
        ticker_data: dict[str, list[tuple[float, float, str]]] = {}  # ticker -> [(score, weight, headline)]

        for article in articles:
            hours_old = max((now - article.published_at).total_seconds() / 3600, 0.01)
            time_weight = math.exp(-decay_lambda * hours_old)
            source_weight = self.source_weights.get(article.source, 0.5)
            conf_weight = article.confidence / 100.0 if article.confidence else 0.5
            total_weight = time_weight * source_weight * conf_weight

            # Direct ticker mentions
            for ticker in article.tickers:
                ticker_data.setdefault(ticker, []).append(
                    (article.sentiment_score, total_weight, article.title)
                )

            # Macro/sector expansion
            sector_impacts = self.check_macro_triggers(article)
            ticker_impacts = self.expand_sector_to_tickers(sector_impacts)
            for ticker, sector_score in ticker_impacts.items():
                # Sector-derived signals get reduced weight
                ticker_data.setdefault(ticker, []).append(
                    (sector_score, total_weight * 0.5, f"[MACRO] {article.title}")
                )

        # Compute weighted average per ticker
        results = {}
        for ticker, entries in ticker_data.items():
            total_weight = sum(w for _, w, _ in entries)
            if total_weight == 0:
                continue
            weighted_score = sum(s * w for s, w, _ in entries) / total_weight
            avg_confidence = min(total_weight * 30, 100)  # Scale weight to confidence
            latest = max(entries, key=lambda x: x[1])  # Highest weight = most recent/credible

            results[ticker] = TickerSentiment(
                ticker=ticker,
                score=round(weighted_score, 2),
                confidence=round(avg_confidence, 1),
                article_count=len(entries),
                latest_headline=latest[2],
            )

        return results
