/**
 * Application configuration
 *
 * Environment variables:
 * - REACT_APP_BACKEND_URL: Backend API URL (defaults to http://localhost:8000 for local dev)
 */

export const config = {
  /**
   * Backend API base URL
   *
   * In production, set REACT_APP_BACKEND_URL to:
   * - Same domain: '' (empty string for relative URLs)
   * - Different domain: 'https://api.example.com'
   */
  backendUrl: process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000',
} as const

export default config
