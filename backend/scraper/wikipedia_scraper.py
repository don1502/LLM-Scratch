"""
Wikipedia scraper for Technology-related articles.
Uses the Wikipedia API to fetch and save article text.
"""

import os
import time
import urllib.request
import urllib.parse
import json
import re


# Technology-related seed topics to scrape
TECHNOLOGY_TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Natural language processing",
    "Computer science",
    "Computer programming",
    "Software engineering",
    "Internet",
    "World Wide Web",
    "Cloud computing",
    "Blockchain",
    "Cryptocurrency",
    "Cybersecurity",
    "Robotics",
    "Quantum computing",
    "5G",
    "Internet of things",
    "Big data",
    "Data science",
    "Algorithm",
    "Operating system",
    "Linux",
    "Python (programming language)",
    "JavaScript",
    "Semiconductor",
    "Microprocessor",
    "Graphics processing unit",
    "Autonomous vehicle",
    "Virtual reality",
    "Augmented reality",
    "Computer vision",
    "Reinforcement learning",
    "Transformer (deep learning architecture)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Generative adversarial network",
    "Transfer learning",
    "Database",
    "SQL",
]


def fetch_wikipedia_article(title):
    """
    Fetch the plain text extract of a Wikipedia article.

    Args:
        title: Wikipedia article title

    Returns:
        Article text as string, or None if not found
    """
    encoded_title = urllib.parse.quote(title)
    url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&titles={encoded_title}"
        f"&prop=extracts&explaintext=1&format=json"
    )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LLM-Scratch-Scraper/1.0 (Educational Project)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")

    except Exception as e:
        print(f"  Error fetching '{title}': {e}")
        return None


def fetch_linked_titles(title, max_links=10):
    """
    Fetch titles of pages linked from a Wikipedia article.

    Args:
        title: Wikipedia article title
        max_links: Maximum number of links to return

    Returns:
        List of linked article titles
    """
    encoded_title = urllib.parse.quote(title)
    url = (
        f"https://en.wikipedia.org/w/api.php?"
        f"action=query&titles={encoded_title}"
        f"&prop=links&pllimit={max_links}&plnamespace=0&format=json"
    )

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "LLM-Scratch-Scraper/1.0 (Educational Project)"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        linked = []
        for page_data in pages.values():
            for link in page_data.get("links", []):
                linked.append(link["title"])
        return linked[:max_links]

    except Exception as e:
        print(f"  Error fetching links for '{title}': {e}")
        return []


def clean_text(text):
    """
    Clean raw Wikipedia text.

    Args:
        text: Raw article text

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove empty sections and excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Remove very short lines (likely headers without content)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 20 or stripped == "":
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return text.strip()


class WikipediaScraper:
    """Scrapes Technology-related articles from Wikipedia."""

    def __init__(self, output_dir="scraped_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def scrape(
        self,
        topics=None,
        max_articles=50,
        links_per_topic=5,
        delay=1.0,
    ):
        """
        Scrape Wikipedia articles on technology topics.

        Args:
            topics: List of topic titles (defaults to TECHNOLOGY_TOPICS)
            max_articles: Maximum total articles to scrape
            links_per_topic: Number of linked articles to follow per topic
            delay: Delay between requests in seconds

        Returns:
            Path to the output file
        """
        if topics is None:
            topics = TECHNOLOGY_TOPICS

        all_text = []
        scraped_titles = set()
        article_count = 0

        print(f"Scraping Wikipedia articles (max {max_articles})...")
        print("=" * 60)

        for topic in topics:
            if article_count >= max_articles:
                break

            if topic in scraped_titles:
                continue

            print(f"\nFetching: {topic}")
            text = fetch_wikipedia_article(topic)
            time.sleep(delay)

            if text:
                cleaned = clean_text(text)
                if len(cleaned) > 100:
                    all_text.append(f"# {topic}\n\n{cleaned}")
                    scraped_titles.add(topic)
                    article_count += 1
                    print(f"  Collected {len(cleaned)} chars")

                    # Follow links
                    if article_count < max_articles:
                        linked = fetch_linked_titles(topic, max_links=links_per_topic)
                        time.sleep(delay)

                        for linked_title in linked:
                            if article_count >= max_articles:
                                break
                            if linked_title in scraped_titles:
                                continue

                            print(f"  -> Fetching linked: {linked_title}")
                            linked_text = fetch_wikipedia_article(linked_title)
                            time.sleep(delay)

                            if linked_text:
                                linked_cleaned = clean_text(linked_text)
                                if len(linked_cleaned) > 100:
                                    all_text.append(
                                        f"# {linked_title}\n\n{linked_cleaned}"
                                    )
                                    scraped_titles.add(linked_title)
                                    article_count += 1
                                    print(f"     Collected {len(linked_cleaned)} chars")

        # Save to file
        output_path = os.path.join(self.output_dir, "technology_corpus.txt")
        corpus = "\n\n---\n\n".join(all_text)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(corpus)

        total_chars = len(corpus)
        print(f"\n{'=' * 60}")
        print(f"Scraping complete!")
        print(f"Articles scraped: {article_count}")
        print(f"Total characters: {total_chars:,}")
        print(f"Output saved to: {output_path}")

        return output_path


if __name__ == "__main__":
    scraper = WikipediaScraper()
    scraper.scrape(max_articles=50, links_per_topic=3, delay=1.0)
