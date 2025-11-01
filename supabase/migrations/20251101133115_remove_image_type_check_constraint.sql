-- Remove the check constraint on image_type column to allow any value
-- This allows for flexibility in image types (extracted, screenshot, diagram, etc.)

ALTER TABLE public.paper_images
DROP CONSTRAINT IF EXISTS paper_images_image_type_check;

-- Optionally, add a comment to document the allowed values
COMMENT ON COLUMN public.paper_images.image_type IS 'Type of image (e.g., extracted, screenshot, chart, figure, diagram, table, other)';
