# Module 1: Data Pipeline

## What I did

Scraped books from books.toscrape.com across 3 categories — Mystery, Historical Fiction, and Fantasy. Ended up with around 120+ books total using pagination.

Used `requests` to fetch pages and `BeautifulSoup` to parse the HTML. Each book had: title, price, star rating, availability, and category.

## Cleaning steps

The raw data needed quite a bit of work:
- Price came as `£12.99` → stripped the £ sign, converted to float
- Rating was a word like "Four" → mapped to integer (One=1, Two=2 ... Five=5)
- Availability was a string like "In stock" → converted to 1/0
- A few books had missing ratings — filled with median rating for that category
- GBP to INR conversion: used fixed rate of **1 GBP = 105.50 INR**

## Database schema

Two tables in SQLite:

```
categories(category_id PK, category_name)
books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)
```

Foreign key constraint is enabled with `PRAGMA foreign_keys = ON`.

## SQL queries

Ran 7 queries covering basic SELECT, WHERE/ORDER BY, GROUP BY aggregations, and a JOIN between books and categories. Some results:

- Average price per category: Mystery ~£12, Fantasy ~£14
- Top rated books: several 5-star books across all 3 categories
- In-stock count: most books were available

## pandas read_sql vs merge

Read the JOIN query result using `pd.read_sql()` and also reproduced it using `pd.merge()` on the two dataframes. Both gave identical output — confirmed with `.equals()` returning True.
