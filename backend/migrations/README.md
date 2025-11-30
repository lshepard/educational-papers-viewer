# Database Migrations

## Running Migrations

### Option 1: Supabase Dashboard (Recommended)

1. Go to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Copy the contents of `create_episode_papers_table.sql`
4. Paste and run the SQL

### Option 2: psql Command Line

```bash
# Using environment variables
psql $DATABASE_URL -f create_episode_papers_table.sql

# Or with explicit connection
psql -h your-project.supabase.co -U postgres -d postgres -f create_episode_papers_table.sql
```

## Migration: Episode Papers Junction Table

**File:** `create_episode_papers_table.sql`

**Purpose:** Creates a many-to-many relationship between podcast episodes and papers.

**Changes:**
- Creates `episode_papers` junction table
- Adds `is_multi_paper` flag to `podcast_episodes`
- Makes `paper_id` nullable in `podcast_episodes` (for multi-paper episodes)

**Why:** Custom themed episodes can include multiple papers from both the database and Semantic Scholar. The junction table allows proper many-to-many relationships while maintaining referential integrity.

## Table Structure

### episode_papers

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| episode_id | UUID | Foreign key to podcast_episodes |
| paper_id | UUID (nullable) | Foreign key to papers table |
| semantic_scholar_id | VARCHAR | Semantic Scholar paper ID |
| paper_title | TEXT | Paper title (denormalized for display) |
| paper_authors | TEXT | Paper authors (denormalized) |
| paper_year | INTEGER | Publication year |
| display_order | INTEGER | Order for display/discussion |
| created_at | TIMESTAMP | Creation timestamp |

**Constraints:**
- Must have either `paper_id` OR `semantic_scholar_id`
- `episode_id` is required
- Cascade delete when episode is deleted
