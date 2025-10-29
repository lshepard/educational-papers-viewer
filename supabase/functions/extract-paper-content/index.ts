// Edge Function to extract sections and images from papers using Gemini
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'
import { GoogleGenerativeAI } from 'https://esm.sh/@google/generative-ai@0.1.3'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ExtractionRequest {
  paper_id: string
}

interface PaperSection {
  section_type: 'introduction' | 'background' | 'methods' | 'results' | 'discussion' | 'conclusion' | 'abstract' | 'other'
  section_title?: string
  content: string
}

interface PaperImage {
  image_type: 'screenshot' | 'chart' | 'figure' | 'diagram' | 'table' | 'other'
  caption?: string
  description?: string
  page_number?: number
}

Deno.serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get environment variables
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const geminiApiKey = Deno.env.get('GEMINI_API_KEY')!

    // Create Supabase client with service role key for admin access
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    // Parse request
    const { paper_id }: ExtractionRequest = await req.json()

    if (!paper_id) {
      throw new Error('paper_id is required')
    }

    console.log(`Starting extraction for paper: ${paper_id}`)

    // Update processing status to 'processing'
    await supabase
      .from('papers')
      .update({ processing_status: 'processing' })
      .eq('id', paper_id)

    // Fetch paper details
    const { data: paper, error: paperError } = await supabase
      .from('papers')
      .select('*')
      .eq('id', paper_id)
      .single()

    if (paperError || !paper) {
      throw new Error(`Failed to fetch paper: ${paperError?.message}`)
    }

    // Get PDF content
    let pdfUrl: string
    if (paper.storage_bucket && paper.storage_path) {
      // Get from Supabase storage
      const { data: urlData } = await supabase.storage
        .from(paper.storage_bucket)
        .createSignedUrl(paper.storage_path, 3600) // 1 hour

      if (!urlData?.signedUrl) {
        throw new Error('Failed to get signed URL for PDF')
      }
      pdfUrl = urlData.signedUrl
    } else if (paper.paper_url) {
      pdfUrl = paper.paper_url
    } else {
      throw new Error('No PDF URL or storage path available')
    }

    console.log(`Fetching PDF from: ${pdfUrl}`)

    // Download PDF
    const pdfResponse = await fetch(pdfUrl)
    if (!pdfResponse.ok) {
      throw new Error(`Failed to fetch PDF: ${pdfResponse.statusText}`)
    }

    const pdfBuffer = await pdfResponse.arrayBuffer()
    const pdfBase64 = btoa(String.fromCharCode(...new Uint8Array(pdfBuffer)))

    console.log(`PDF downloaded, size: ${pdfBuffer.byteLength} bytes`)

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(geminiApiKey)
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-pro' })

    // Create extraction prompt
    const prompt = `
You are a research paper analyzer. Extract the following information from this academic paper:

1. **Paper Sections**: Identify and extract these sections (if present):
   - Abstract
   - Introduction
   - Background/Related Work
   - Methods/Methodology
   - Results
   - Discussion
   - Conclusion

2. **Images and Figures**: Identify all images, charts, diagrams, screenshots, and tables. For each:
   - Type (screenshot, chart, figure, diagram, table)
   - Caption (if present)
   - Brief description of what the image shows
   - Page number (if identifiable)

Return your response as a JSON object with this structure:
{
  "sections": [
    {
      "section_type": "introduction|background|methods|results|discussion|conclusion|abstract|other",
      "section_title": "optional title",
      "content": "full text content of this section"
    }
  ],
  "images": [
    {
      "image_type": "screenshot|chart|figure|diagram|table|other",
      "caption": "optional caption",
      "description": "description of what the image shows",
      "page_number": optional_number
    }
  ]
}

Be thorough and extract ALL content. For methods section, be especially detailed.
If you cannot identify a clear section type, use "other" with an appropriate section_title.
`

    console.log('Sending to Gemini for analysis...')

    // Call Gemini with PDF
    const result = await model.generateContent([
      prompt,
      {
        inlineData: {
          mimeType: 'application/pdf',
          data: pdfBase64,
        },
      },
    ])

    const response = await result.response
    const text = response.text()

    console.log('Gemini response received')

    // Parse JSON response
    const jsonMatch = text.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      throw new Error('Failed to parse Gemini response as JSON')
    }

    const extracted = JSON.parse(jsonMatch[0])
    const sections: PaperSection[] = extracted.sections || []
    const images: PaperImage[] = extracted.images || []

    console.log(`Extracted ${sections.length} sections and ${images.length} images`)

    // Store sections in database
    if (sections.length > 0) {
      const sectionsToInsert = sections.map((section) => ({
        paper_id,
        section_type: section.section_type,
        section_title: section.section_title,
        content: section.content,
      }))

      const { error: sectionsError } = await supabase
        .from('paper_sections')
        .insert(sectionsToInsert)

      if (sectionsError) {
        console.error('Failed to insert sections:', sectionsError)
        throw sectionsError
      }
    }

    // Store images in database
    if (images.length > 0) {
      const imagesToInsert = images.map((image) => ({
        paper_id,
        image_type: image.image_type,
        caption: image.caption,
        description: image.description,
        page_number: image.page_number,
      }))

      const { error: imagesError } = await supabase
        .from('paper_images')
        .insert(imagesToInsert)

      if (imagesError) {
        console.error('Failed to insert images:', imagesError)
        throw imagesError
      }
    }

    // Update processing status to 'completed'
    await supabase
      .from('papers')
      .update({
        processing_status: 'completed',
        processed_at: new Date().toISOString(),
      })
      .eq('id', paper_id)

    console.log('Extraction completed successfully')

    return new Response(
      JSON.stringify({
        success: true,
        paper_id,
        sections_count: sections.length,
        images_count: images.length,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )
  } catch (error) {
    console.error('Extraction error:', error)

    // Try to update paper status to failed
    try {
      const supabaseUrl = Deno.env.get('SUPABASE_URL')!
      const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
      const supabase = createClient(supabaseUrl, supabaseServiceKey)

      const { paper_id } = await req.json()
      if (paper_id) {
        await supabase
          .from('papers')
          .update({
            processing_status: 'failed',
            processing_error: error.message,
          })
          .eq('id', paper_id)
      }
    } catch (e) {
      console.error('Failed to update error status:', e)
    }

    return new Response(
      JSON.stringify({
        success: false,
        error: error.message,
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500,
      }
    )
  }
})
