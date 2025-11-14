import React, { useState, useCallback } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import ReactMarkdown from 'react-markdown'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'
import { GenaiPaper, supabase } from '../supabase'
import { useAuth } from '../contexts/AuthContext'
import PaperUploadEdit from './PaperUploadEdit'

// Set up the worker for react-pdf - use matching version from unpkg
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

interface PDFViewerProps {
  paper: GenaiPaper
  onClose: () => void
}

interface PaperImage {
  id: string
  paper_id: string
  page_number: number | null
  image_type: string
  storage_path: string
  width: number | null
  height: number | null
  created_at: string
}

type ViewTab = 'content' | 'images';

const PDFViewer: React.FC<PDFViewerProps> = ({ paper, onClose }) => {
  const [numPages, setNumPages] = useState<number>(0)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [markdownContent, setMarkdownContent] = useState<string | null>(null)
  const [showUploadEdit, setShowUploadEdit] = useState<boolean>(false)
  const [images, setImages] = useState<PaperImage[]>([])
  const [imagesLoading, setImagesLoading] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<ViewTab>('content')
  const [extracting, setExtracting] = useState<boolean>(false)
  const [extractionMessage, setExtractionMessage] = useState<string | null>(null)
  const { user } = useAuth()

  const loadContent = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      if (paper.file_kind === 'pdf') {
        if (paper.storage_bucket && paper.storage_path) {
          // Get PDF from Supabase storage
          // Remove bucket name from storage_path if it's already included
          let cleanPath = paper.storage_path
          if (cleanPath.startsWith(`${paper.storage_bucket}/`)) {
            cleanPath = cleanPath.substring(`${paper.storage_bucket}/`.length)
          }

          const { data } = await supabase.storage
            .from(paper.storage_bucket)
            .getPublicUrl(cleanPath)

          if (data?.publicUrl) {
            setPdfUrl(data.publicUrl)
          } else {
            setPdfUrl(paper.paper_url || paper.source_url)
          }
        } else {
          // Use the paper URL directly
          setPdfUrl(paper.paper_url || paper.source_url)
        }
      } else if (paper.file_kind === 'markdown') {
        // Load markdown content from database
        if (paper.markdown) {
          setMarkdownContent(paper.markdown)
        } else {
          setError('No markdown content available for this paper')
        }
      } else {
        setError(`Unsupported file type: ${paper.file_kind}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load content')
    } finally {
      setLoading(false)
    }
  }, [paper])

  const loadImages = useCallback(async () => {
    setImagesLoading(true)

    try {
      const { data, error } = await supabase
        .from('paper_images')
        .select('*')
        .eq('paper_id', paper.id)
        .order('page_number', { ascending: true })

      if (error) throw error

      setImages(data || [])
    } catch (err) {
      console.error('Failed to load images:', err)
      // Don't set error state, just fail silently for images
    } finally {
      setImagesLoading(false)
    }
  }, [paper.id])

  React.useEffect(() => {
    loadContent()
    loadImages()
  }, [loadContent, loadImages])

  const getImageUrl = (image: PaperImage): string => {
    const bucket = paper.storage_bucket || 'papers'
    const { data } = supabase.storage.from(bucket).getPublicUrl(image.storage_path)
    return data.publicUrl
  }

  const handleExtractContent = async () => {
    setExtracting(true)
    setExtractionMessage(null)

    try {
      const response = await fetch('http://localhost:8000/extract', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          paper_id: paper.id,
        }),
      })

      if (!response.ok) {
        throw new Error('Extraction failed')
      }

      const result = await response.json()
      setExtractionMessage(`✓ Extracted ${result.sections_count} sections and ${result.images_count} images`)

      // Reload images after extraction
      await loadImages()

      // Switch to images tab if images were extracted
      if (result.images_count > 0) {
        setActiveTab('images')
      }
    } catch (err) {
      setExtractionMessage(`✗ Extraction failed: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setExtracting(false)
    }
  }

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
  }

  const onDocumentLoadError = (error: Error) => {
    setError(`Failed to load PDF: ${error.message}`)
  }

  // Create array of page numbers for rendering all pages
  const renderAllPages = () => {
    if (numPages === 0) return null
    
    return Array.from({ length: numPages }, (_, index) => (
      <div key={index + 1} className="pdf-page-container">
        <div className="pdf-page-number">Page {index + 1}</div>
        <Page 
          pageNumber={index + 1} 
          renderTextLayer={false}
          renderAnnotationLayer={false}
          className="pdf-page"
          width={Math.min(800, window.innerWidth * 0.6)}
        />
      </div>
    ))
  }

  if (loading) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-header">
          <button onClick={onClose} className="close-btn">← Back</button>
          <h2>Loading {paper.file_kind === 'markdown' ? 'content' : 'PDF'}...</h2>
        </div>
      </div>
    )
  }

  const handleUploadSuccess = () => {
    setShowUploadEdit(false)
    setError(null)
    // Reload the content
    loadContent()
  }

  if (error) {
    return (
      <div className="pdf-viewer">
        <div className="pdf-header">
          <button onClick={onClose} className="close-btn">← Back</button>
          <h2>Error</h2>
        </div>
        <div className="error">{error}</div>

        {user && !showUploadEdit && (
          <div className="admin-action-container">
            <button
              onClick={() => setShowUploadEdit(true)}
              className="fix-pdf-btn"
            >
              Upload PDF or Fix URL
            </button>
          </div>
        )}

        {showUploadEdit && user && (
          <PaperUploadEdit
            paper={paper}
            onSuccess={handleUploadSuccess}
            onCancel={() => setShowUploadEdit(false)}
          />
        )}

        <div className="paper-info">
          <h3>Paper Information</h3>
          <p><strong>Title:</strong> {paper.title || 'Untitled'}</p>
          <p><strong>URL:</strong> <a href={paper.source_url} target="_blank" rel="noopener noreferrer">{paper.source_url}</a></p>
          <p><strong>File Type:</strong> {paper.file_kind || 'other'}</p>
          <p><strong>Authors:</strong> {paper.authors || 'Unknown'}</p>
          <p><strong>Year:</strong> {paper.year || 'Unknown'}</p>
          <p><strong>Venue:</strong> {paper.venue || 'Unknown'}</p>
          {paper.application && <p><strong>Application:</strong> {paper.application}</p>}
          {paper.users && <p><strong>Users:</strong> {paper.users}</p>}
          {paper.ages && <p><strong>Ages:</strong> {paper.ages}</p>}
        </div>
      </div>
    )
  }

  return (
    <div className="pdf-viewer-layout">
      {/* Header */}
      <div className="pdf-header">
        <button onClick={onClose} className="close-btn">← Back to Papers</button>
        <div className="pdf-info">
          {paper.file_kind === 'pdf' && numPages > 0 && (
            <span className="page-count">{numPages} pages</span>
          )}
          {paper.file_kind === 'markdown' && (
            <span className="page-count">Markdown</span>
          )}
          {extractionMessage && (
            <span className={`extraction-message ${extractionMessage.startsWith('✓') ? 'success' : 'error'}`}>
              {extractionMessage}
            </span>
          )}
        </div>
        {user && !showUploadEdit && (
          <div className="header-actions">
            <button
              onClick={handleExtractContent}
              className="extract-content-btn"
              disabled={extracting}
            >
              {extracting ? 'Extracting...' : 'Extract Content'}
            </button>
            <button
              onClick={() => setShowUploadEdit(true)}
              className="change-pdf-btn"
            >
              Change PDF
            </button>
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      {!showUploadEdit && (
        <div className="viewer-tabs">
          <button
            className={`viewer-tab-button ${activeTab === 'content' ? 'active' : ''}`}
            onClick={() => setActiveTab('content')}
          >
            {paper.file_kind === 'pdf' ? 'PDF' : 'Content'}
          </button>
          {images.length > 0 && (
            <button
              className={`viewer-tab-button ${activeTab === 'images' ? 'active' : ''}`}
              onClick={() => setActiveTab('images')}
            >
              Images ({images.length})
            </button>
          )}
        </div>
      )}

      {/* Upload/Edit form when admin wants to change PDF */}
      {showUploadEdit && user && (
        <div className="pdf-main-content">
          <PaperUploadEdit
            paper={paper}
            onSuccess={handleUploadSuccess}
            onCancel={() => setShowUploadEdit(false)}
          />
        </div>
      )}

      {/* Main content with sidebar and PDF */}
      {!showUploadEdit && (
        <div className="pdf-main-content">
          {/* Left sidebar with paper details */}
          <div className="pdf-sidebar">
          <div className="paper-metadata">
            <h2 className="paper-title">{paper.title || 'Untitled Paper'}</h2>
            
            <div className="metadata-section">
              <h3>Authors</h3>
              <p>{paper.authors || 'Unknown'}</p>
            </div>

            <div className="metadata-section">
              <h3>Publication</h3>
              <p><strong>Year:</strong> {paper.year || 'Unknown'}</p>
              <p><strong>Venue:</strong> {paper.venue || 'Unknown'}</p>
            </div>

            {paper.application && (
              <div className="metadata-section">
                <h3>Application</h3>
                <p>{paper.application}</p>
              </div>
            )}

            {paper.users && (
              <div className="metadata-section">
                <h3>Target Users</h3>
                <p>{paper.users}</p>
              </div>
            )}

            {paper.ages && (
              <div className="metadata-section">
                <h3>Age Groups</h3>
                <p>{paper.ages}</p>
              </div>
            )}

            {paper.study_design && (
              <div className="metadata-section">
                <h3>Study Design</h3>
                <p>{paper.study_design}</p>
              </div>
            )}

            {paper.why && (
              <div className="metadata-section">
                <h3>Purpose</h3>
                <p>{paper.why}</p>
              </div>
            )}

            <div className="metadata-section">
              <h3>Links</h3>
              <p>
                <a href={paper.source_url} target="_blank" rel="noopener noreferrer" className="external-link">
                  View Original Source
                </a>
              </p>
            </div>

            {/* Image count info */}
            {images.length > 0 && (
              <div className="metadata-section">
                <h3>Extracted Images</h3>
                <p>{images.length} images extracted</p>
                <button
                  onClick={() => setActiveTab('images')}
                  className="view-images-btn"
                >
                  View All Images
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right side content viewer */}
        {activeTab === 'content' && (
          <div className="pdf-content-area">
            {paper.file_kind === 'pdf' && pdfUrl && (
              <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                onLoadError={onDocumentLoadError}
                loading={<div className="pdf-loading">Loading PDF...</div>}
              >
                <div className="pdf-pages-container">
                  {renderAllPages()}
                </div>
              </Document>
            )}

            {paper.file_kind === 'markdown' && markdownContent && (
              <div className="markdown-content">
                <ReactMarkdown>{markdownContent}</ReactMarkdown>
              </div>
            )}
          </div>
        )}

        {/* Images View */}
        {activeTab === 'images' && (
          <div className="images-full-view">
            <div className="images-grid-container">
              {imagesLoading ? (
                <div className="loading">Loading images...</div>
              ) : images.length === 0 ? (
                <div className="no-results">No images extracted from this paper.</div>
              ) : (
                <div className="images-grid">
                  {images.map((image) => (
                    <div key={image.id} className="image-card">
                      <img
                        src={getImageUrl(image)}
                        alt={`Page ${image.page_number || '?'}`}
                        className="image-card-img"
                      />
                      <div className="image-card-info">
                        <div className="image-card-meta">
                          <span className="image-page-badge">Page {image.page_number || '?'}</span>
                          <span className={`image-type-badge ${image.image_type}`}>
                            {image.image_type === 'vector' ? 'Figure/Chart' : 'Embedded'}
                          </span>
                        </div>
                        {image.width && image.height && (
                          <div className="image-dimensions-text">{image.width} × {image.height}px</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
      )}
    </div>
  )
}

export default PDFViewer