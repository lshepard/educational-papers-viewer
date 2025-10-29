# Extract Paper Content Edge Function

This Supabase Edge Function extracts sections and images from academic papers using Google's Gemini 2.5 Pro AI model.

## Features

- Extracts paper sections: Introduction, Background, Methods, Results, Discussion, Conclusion
- Identifies and catalogs images, charts, figures, and tables
- Stores extracted content in `paper_sections` and `paper_images` tables
- Updates paper processing status

## Environment Variables

Required environment variables (set in Supabase Dashboard):

- `SUPABASE_URL` - Your Supabase project URL (automatically provided)
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key (automatically provided)
- `GEMINI_API_KEY` - Your Google Gemini API key

## Request Format

```json
{
  "paper_id": "uuid-of-paper"
}
```

## Response Format

Success:
```json
{
  "success": true,
  "paper_id": "uuid",
  "sections_count": 7,
  "images_count": 12
}
```

Error:
```json
{
  "success": false,
  "error": "error message"
}
```

## Deployment

Deploy this function using:

```bash
npx supabase functions deploy extract-paper-content
```

## Setting Environment Variables

```bash
npx supabase secrets set GEMINI_API_KEY=your_gemini_api_key_here
```

## Testing Locally

```bash
npx supabase functions serve extract-paper-content
```

Then call with:
```bash
curl -i --location --request POST 'http://localhost:54321/functions/v1/extract-paper-content' \
  --header 'Authorization: Bearer YOUR_ANON_KEY' \
  --header 'Content-Type: application/json' \
  --data '{"paper_id":"your-paper-uuid"}'
```
