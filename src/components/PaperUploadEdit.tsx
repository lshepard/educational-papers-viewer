import React, { useState } from 'react'
import { GenaiPaper, supabase } from '../supabase'

interface PaperUploadEditProps {
  paper: GenaiPaper
  onSuccess: () => void
  onCancel: () => void
}

const PaperUploadEdit: React.FC<PaperUploadEditProps> = ({ paper, onSuccess, onCancel }) => {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [pdfUrl, setPdfUrl] = useState(paper.paper_url || '')
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      if (file.type === 'application/pdf') {
        setPdfFile(file)
        setError(null)
      } else {
        setError('Please select a PDF file')
        setPdfFile(null)
      }
    }
  }

  const handleUploadPDF = async () => {
    if (!pdfFile) {
      setError('Please select a PDF file')
      return
    }

    setUploading(true)
    setError(null)
    setSuccessMessage(null)

    try {
      // Generate filename from paper ID or title
      const timestamp = Date.now()
      const sanitizedTitle = paper.title?.replace(/[^a-z0-9]/gi, '-').toLowerCase() || 'paper'
      const filename = `${sanitizedTitle}-${timestamp}.pdf`
      const filePath = filename

      // Upload to Supabase storage
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('papers')
        .upload(filePath, pdfFile, {
          cacheControl: '3600',
          upsert: false
        })

      if (uploadError) throw uploadError

      // Update the database record
      const { error: updateError } = await supabase
        .from('genai_papers')
        .update({
          storage_bucket: 'papers',
          storage_path: uploadData.path,
          file_kind: 'pdf'
        })
        .eq('id', paper.id)

      if (updateError) throw updateError

      setSuccessMessage('PDF uploaded successfully!')
      setTimeout(() => {
        onSuccess()
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload PDF')
    } finally {
      setUploading(false)
    }
  }

  const handleUpdateUrl = async () => {
    if (!pdfUrl.trim()) {
      setError('Please enter a PDF URL')
      return
    }

    setUploading(true)
    setError(null)
    setSuccessMessage(null)

    try {
      const { error: updateError } = await supabase
        .from('genai_papers')
        .update({
          paper_url: pdfUrl,
          file_kind: 'pdf'
        })
        .eq('id', paper.id)

      if (updateError) throw updateError

      setSuccessMessage('PDF URL updated successfully!')
      setTimeout(() => {
        onSuccess()
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update URL')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="paper-upload-edit">
      <div className="upload-edit-header">
        <h3>Upload or Update PDF</h3>
        <button onClick={onCancel} className="close-upload-btn">×</button>
      </div>

      <div className="upload-edit-content">
        {/* Upload PDF File Section */}
        <div className="upload-section">
          <h4>Option 1: Upload PDF File</h4>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="file-input"
            disabled={uploading}
          />
          {pdfFile && (
            <p className="file-selected">Selected: {pdfFile.name}</p>
          )}
          <button
            onClick={handleUploadPDF}
            disabled={!pdfFile || uploading}
            className="upload-btn"
          >
            {uploading ? 'Uploading...' : 'Upload PDF'}
          </button>
        </div>

        <div className="divider">
          <span>OR</span>
        </div>

        {/* Update PDF URL Section */}
        <div className="url-section">
          <h4>Option 2: Update PDF URL</h4>
          <input
            type="url"
            value={pdfUrl}
            onChange={(e) => setPdfUrl(e.target.value)}
            placeholder="https://example.com/paper.pdf"
            className="url-input"
            disabled={uploading}
          />
          <button
            onClick={handleUpdateUrl}
            disabled={!pdfUrl.trim() || uploading}
            className="update-url-btn"
          >
            {uploading ? 'Updating...' : 'Update URL'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {successMessage && (
          <div className="success-message">
            {successMessage}
          </div>
        )}
      </div>
    </div>
  )
}

export default PaperUploadEdit
