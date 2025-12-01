import React from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Link, useNavigate } from 'react-router-dom'

const AdminDashboard: React.FC = () => {
  const { user, signOut, loading } = useAuth()
  const navigate = useNavigate()

  // If not logged in, redirect to login (but wait for auth to finish loading)
  React.useEffect(() => {
    if (!loading && !user) {
      navigate('/admin/login')
    }
  }, [user, loading, navigate])

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  // Show loading state while auth is initializing
  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        fontSize: '1.2rem',
        color: '#666'
      }}>
        Loading...
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <div className="admin-dashboard">
      <div className="admin-dashboard-header">
        <h1>Admin Dashboard</h1>
        <div className="admin-dashboard-actions">
          <button onClick={() => navigate('/')} className="home-btn">
            ← Back to Home
          </button>
          <button onClick={handleSignOut} className="sign-out-btn">
            Sign Out
          </button>
        </div>
      </div>

      <div className="admin-dashboard-content">
        <div className="admin-card">
          <h2>📄 Import Paper</h2>
          <p>Add papers from arXiv or direct PDF links</p>
          <Link to="/admin/import" className="admin-card-button">
            Import Paper
          </Link>
        </div>

        <div className="admin-card">
          <h2>⚙️ Paper Processing</h2>
          <p>Extract content from uploaded papers</p>
          <Link to="/admin/processing" className="admin-card-button">
            Go to Processing
          </Link>
        </div>

        <div className="admin-card">
          <h2>🎙️ Podcast Manager</h2>
          <p>View, edit, and regenerate podcast episodes</p>
          <Link to="/admin/podcast-manager" className="admin-card-button">
            Manage Podcasts
          </Link>
        </div>
      </div>
    </div>
  )
}

export default AdminDashboard
