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

  // Parse request body outside try/catch so we can access paper_id in error handler
  let paper_id: string | undefined

  try {
    // Get environment variables
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    const geminiApiKey = Deno.env.get('GEMINI_API_KEY')!

    // Create Supabase client with service role key for admin access
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    // Parse request
    const requestBody: ExtractionRequest = await req.json()
    paper_id = requestBody.paper_id

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
      // Get public URL from Supabase storage (bucket has public read access)
      // Strip bucket name prefix if it exists in the path (e.g., 'papers/file.pdf' -> 'file.pdf')
      let cleanPath = paper.storage_path
      const bucketPrefix = `${paper.storage_bucket}/`
      if (cleanPath.startsWith(bucketPrefix)) {
        cleanPath = cleanPath.substring(bucketPrefix.length)
      }

      const { data: urlData } = supabase.storage
        .from(paper.storage_bucket)
        .getPublicUrl(cleanPath)

      if (!urlData?.publicUrl) {
        throw new Error('Failed to get public URL for PDF')
      }
      pdfUrl = urlData.publicUrl
      console.log(`Using storage URL: ${pdfUrl} (path: ${cleanPath})`)
    } else if (paper.paper_url) {
      pdfUrl = paper.paper_url
      console.log(`Using external URL: ${pdfUrl}`)
    } else {
      throw new Error('No PDF URL or storage path available')
    }

    console.log(`Fetching PDF from: ${pdfUrl}`)

    // Download PDF
    const pdfResponse = await fetch(pdfUrl)
    if (!pdfResponse.ok) {
      throw new Error(`Failed to fetch PDF: ${pdfResponse.statusText}`)
    }

    const pdfArrayBuffer = await pdfResponse.arrayBuffer()
    const numBytes = pdfArrayBuffer.byteLength
    console.log(`PDF downloaded, size: ${numBytes} bytes`)

    // Step 1: Start resumable upload
    console.log('Starting resumable upload to Google Files API...')
    const uploadUrl = `https://generativelanguage.googleapis.com/upload/v1beta/files`

    const startResponse = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'X-Goog-Upload-Protocol': 'resumable',
        'X-Goog-Upload-Command': 'start',
        'X-Goog-Upload-Header-Content-Length': numBytes.toString(),
        'X-Goog-Upload-Header-Content-Type': 'application/pdf',
        'Content-Type': 'application/json',
        'x-goog-api-key': geminiApiKey,
      },
      body: JSON.stringify({
        file: {
          display_name: paper.title || 'Research Paper'
        }
      })
    })

    if (!startResponse.ok) {
      const errorText = await startResponse.text()
      throw new Error(`Failed to start upload: ${startResponse.statusText} - ${errorText}`)
    }

    const uploadLocation = startResponse.headers.get('x-goog-upload-url')
    if (!uploadLocation) {
      throw new Error('No upload URL received from start request')
    }

    console.log(`Upload URL received, uploading file...`)

    // Step 2: Upload the file bytes
    const uploadResponse = await fetch(uploadLocation, {
      method: 'POST',
      headers: {
        'Content-Length': numBytes.toString(),
        'X-Goog-Upload-Offset': '0',
        'X-Goog-Upload-Command': 'upload, finalize',
      },
      body: pdfArrayBuffer,
    })

    if (!uploadResponse.ok) {
      const errorText = await uploadResponse.text()
      throw new Error(`Failed to upload file: ${uploadResponse.statusText} - ${errorText}`)
    }

    const uploadResult = await uploadResponse.json()
    const fileUri = uploadResult.file.uri
    console.log(`File uploaded successfully: ${fileUri}`)

    // Initialize Gemini
    const genAI = new GoogleGenerativeAI(geminiApiKey)
    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-pro' })

    // ========== EXTRACTION 1: Text Sections ==========
    console.log('Extracting text sections from PDF...')
    const sectionsPrompt = `
Extract all text sections from this research paper PDF.

Return ONLY valid JSON with this structure (no markdown, no explanations):
{"sections":[{"section_type":"abstract","section_title":"Abstract","content":"full text here"},...]}

Extract these sections if present:
- Abstract (section_type: "abstract")
- Introduction (section_type: "introduction")
- Background/Related Work (section_type: "background")
- Methods/Methodology (section_type: "methods") - BE VERY DETAILED HERE
- Results (section_type: "results")
- Discussion (section_type: "discussion")
- Conclusion (section_type: "conclusion")
- Any other sections (section_type: "other")

Include the COMPLETE text content for each section. For Methods, extract every detail about the methodology.
`

    const sectionsResult = await model.generateContent([
      { text: sectionsPrompt },
      { file_data: { mime_type: 'application/pdf', file_uri: fileUri } }
    ])

    let sectionsText = sectionsResult.response.text()
    sectionsText = sectionsText.replace(/```json\n?/g, '').replace(/```\n?/g, '')
    const sectionsJsonMatch = sectionsText.match(/\{[\s\S]*\}/)

    let sections: PaperSection[] = []
    if (sectionsJsonMatch) {
      try {
        const sectionsData = JSON.parse(sectionsJsonMatch[0])
        sections = sectionsData.sections || []
        console.log(`Extracted ${sections.length} sections`)
      } catch (e) {
        console.error('Failed to parse sections JSON:', e)
      }
    }

    // ========== EXTRACTION 2: Images and Figures ==========
    console.log('Extracting images and figures from PDF...')
    const imagesPrompt = `
Identify and describe ALL images, figures, charts, diagrams, screenshots, and tables in this research paper PDF.

Return ONLY valid JSON with this structure (no markdown, no explanations):
{"images":[{"image_type":"figure","caption":"Figure 1: ...","description":"detailed description","page_number":1},...]}

For each visual element:
- image_type: choose from screenshot, chart, figure, diagram, table, other
- caption: extract the exact caption if present
- description: describe what the image shows in detail
- page_number: the page it appears on (if identifiable)

Be thorough - identify EVERY visual element in the paper.
`

    const imagesResult = await model.generateContent([
      { text: imagesPrompt },
      { file_data: { mime_type: 'application/pdf', file_uri: fileUri } }
    ])

    let imagesText = imagesResult.response.text()
    imagesText = imagesText.replace(/```json\n?/g, '').replace(/```\n?/g, '')
    const imagesJsonMatch = imagesText.match(/\{[\s\S]*\}/)

    let images: PaperImage[] = []
    if (imagesJsonMatch) {
      try {
        const imagesData = JSON.parse(imagesJsonMatch[0])
        images = imagesData.images || []
        console.log(`Extracted ${images.length} images`)
      } catch (e) {
        console.error('Failed to parse images JSON:', e)
      }
    }

    console.log(`Total extracted: ${sections.length} sections and ${images.length} images`)

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

    // Try to update paper status to failed (paper_id was parsed earlier)
    if (paper_id) {
      try {
        const supabaseUrl = Deno.env.get('SUPABASE_URL')!
        const supabaseServiceKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
        const supabase = createClient(supabaseUrl, supabaseServiceKey)

        await supabase
          .from('papers')
          .update({
            processing_status: 'failed',
            processing_error: error.message,
          })
          .eq('id', paper_id)
      } catch (e) {
        console.error('Failed to update error status:', e)
      }
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
