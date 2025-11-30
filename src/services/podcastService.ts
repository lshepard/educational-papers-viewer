import config from '../config'

export interface PodcastEpisode {
  id: string
  paper_id: string
  title: string
  description: string | null
  duration_seconds: number | null
  storage_bucket: string
  storage_path: string | null
  audio_url: string | null
  autocontent_request_id: string
  generation_status: 'pending' | 'processing' | 'downloading' | 'completed' | 'failed'
  generation_error: string | null
  published_at: string
  episode_number: number | null
  season_number: number | null
  explicit: boolean
  created_at: string
  updated_at: string
}

export interface PodcastGenerationResponse {
  success: boolean
  episode_id: string
  audio_url: string
  message: string
}

export class PodcastService {
  /**
   * Generate a podcast episode from a research paper using Gemini + Google TTS
   *
   * This operation is synchronous and takes 2-5 minutes to complete.
   * The response will include the audio URL when generation is complete.
   *
   * @param paperId - The ID of the paper to generate a podcast from
   * @returns Response with audio URL
   */
  static async generatePodcast(paperId: string): Promise<PodcastGenerationResponse> {
    try {
      const response = await fetch(`${config.backendUrl}/podcast/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ paper_id: paperId }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to generate podcast')
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to generate podcast:', error)
      throw error
    }
  }

  /**
   * Get podcast episode for a specific paper
   *
   * @param paperId - The ID of the paper
   * @returns The podcast episode if it exists, null otherwise
   */
  static async getEpisodeByPaperId(paperId: string): Promise<PodcastEpisode | null> {
    try {
      const response = await fetch(`${config.backendUrl}/podcast/episodes`)
      if (!response.ok) {
        return null
      }

      const data = await response.json()
      if (data.success && data.episodes) {
        const episode = data.episodes.find((ep: PodcastEpisode) => ep.paper_id === paperId)
        return episode || null
      }

      return null
    } catch (error) {
      console.error('Failed to fetch podcast episode:', error)
      return null
    }
  }

  /**
   * Get the podcast RSS feed URL
   *
   * @returns The URL to the RSS feed
   */
  static getRssFeedUrl(): string {
    return `${config.backendUrl}/podcast/feed.xml`
  }
}
