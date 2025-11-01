-- Add width and height columns to paper_images table for storing image dimensions

ALTER TABLE public.paper_images
ADD COLUMN IF NOT EXISTS width integer,
ADD COLUMN IF NOT EXISTS height integer;

-- Add comments to document the columns
COMMENT ON COLUMN public.paper_images.width IS 'Width of the extracted image in pixels';
COMMENT ON COLUMN public.paper_images.height IS 'Height of the extracted image in pixels';
