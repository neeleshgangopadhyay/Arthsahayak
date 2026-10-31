import feedparser
FEEDS = [
    "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
    "https://www.livemint.com/rss/economy",
    "https://www.moneycontrol.com/rss/economy.xml",
]


def fetch_recent_news(max_items=8):
    """
    Fetches recent headlines from a few Indian economy news RSS feeds.
    Returns a list of dicts: {title, summary, link, source, published}.
    Feeds that fail to load are silently skipped (RSS feeds go down or
    change URLs over time — one bad feed shouldn't break the whole tool).
    """
    items = []
    for url in FEEDS:
        try:
            parsed = feedparser.parse(url)
            if parsed.bozo and not parsed.entries:
                # bozo=True means the feed had a parsing problem; if we got
                # no entries at all, treat this feed as unavailable and move on
                continue
            source_name = parsed.feed.get("title", url)
            for entry in parsed.entries[:max_items]:
                items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                    "link": entry.get("link", ""),
                    "source": source_name,
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            print(f"[news_tool] Warning: feed failed ({url}): {e}")
            continue
    return items


if __name__ == "__main__":
    news = fetch_recent_news()
    print(f"Fetched {len(news)} items from {len(FEEDS)} feeds.\n")
    by_source = {}
    for item in news:
        by_source.setdefault(item["source"], 0)
        by_source[item["source"]] += 1
    print("Items per source:")
    for source, count in by_source.items():
        print(f"  {source}: {count}")
    print()
    for item in news[:5]:
        print(f"[{item['source']}] {item['title']}")
        print(f"  {item['link']}")
        print()