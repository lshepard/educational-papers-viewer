# API Endpoint Audit

## Currently Used by Frontend

### Papers
- ✅ `POST /extract` - Extract content from a paper (PaperProcessing, PaperImport)
- ✅ `POST /papers/import` - Import paper from URL (PaperImport)
- ✅ `GET /papers/search?q=` - Search internal papers DB (CustomEpisodeCreator)

### Semantic Scholar
- ✅ `GET /semantic-scholar/search?q=` - Search SS for papers (CustomEpisodeCreator)
- ✅ `POST /semantic-scholar/citations` - Get citations for papers (CustomEpisodeCreator)

### Podcasts
- ✅ `POST /podcast/generate` - Generate single-paper podcast (podcastService.ts)
- ✅ `POST /podcast/generate-custom` - Generate multi-paper podcast (CustomEpisodeCreator)
- ✅ `GET /podcast/episodes` - List all episodes (podcastService.ts)
- ✅ `GET /podcast/feed.xml` - RSS feed (podcastService.ts)

## Available But NOT Used

### Papers
- ❌ `POST /papers/batch-extract` - Batch extract papers
- ❌ `GET /papers/processing-stats` - Get processing statistics

### Podcasts
- ❌ `GET /podcast/episodes/{id}` - Get single episode
- ❌ `PATCH /podcast/episodes/{id}` - Update episode metadata
- ❌ `DELETE /podcast/episodes/{id}` - Delete episode
- ❌ `POST /podcast/episodes/{id}/regenerate-audio` - Regenerate audio from script

### Search
- ❌ `POST /search/papers` - Full-text search paper sections
- ❌ `POST /search/test` - Test search functionality

### Admin
- ❌ `POST /admin/populate-research` - Populate research metadata

## Issues to Fix

### 1. Inconsistent Endpoints
- `/extract` should be `/papers/extract`
- `/semantic-scholar/*` should be `/search/semantic-scholar/*`

### 2. Duplicate Podcast Generation
- Should have ONE endpoint for podcast generation (single or multi-paper)
- Current: `/podcast/generate` (single) AND `/podcast/generate-custom` (multi-paper)
- Proposed: `/podcast/generate` accepts `paper_ids: string[]` (1 or many)

### 3. Search Confusion
- `/papers/search` (searches DB directly)
- `/search/papers` (full-text search via backend)
- Should consolidate

### 4. Unused CRUD Operations
- Episode management endpoints (GET/PATCH/DELETE single) not used
- Either expose in UI or remove

## Proposed Clean API

```
Papers:
  POST   /papers/extract          - Extract single paper
  POST   /papers/batch-extract    - Extract multiple papers
  POST   /papers/import           - Import from URL
  GET    /papers/stats            - Processing statistics

Search:
  GET    /search?q=               - Search internal papers
  GET    /search/semantic-scholar?q= - Search Semantic Scholar
  POST   /search/citations        - Get citations for papers

Podcasts:
  POST   /podcasts                - Generate episode (1+ papers)
  GET    /podcasts                - List episodes
  GET    /podcasts/:id            - Get episode
  PATCH  /podcasts/:id            - Update episode
  DELETE /podcasts/:id            - Delete episode
  GET    /podcasts/feed.xml       - RSS feed

Admin:
  POST   /admin/research/populate - Populate research metadata
```

## Next Steps

1. Update routers to use clean URL structure
2. Consolidate podcast generation into single endpoint
3. Update frontend to match new endpoints
4. Remove unused endpoints or add UI for them
5. Document final API in OpenAPI spec
