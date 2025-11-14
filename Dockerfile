# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies for building)
RUN npm install

# Copy source code
COPY . .

# Accept build arguments from Railway
ARG REACT_APP_SUPABASE_URL
ARG REACT_APP_SUPABASE_ANON_KEY

# Set them as environment variables for the build
ENV REACT_APP_SUPABASE_URL=$REACT_APP_SUPABASE_URL
ENV REACT_APP_SUPABASE_ANON_KEY=$REACT_APP_SUPABASE_ANON_KEY

# Build the React app
RUN npm run build

# Production stage  
FROM node:20-alpine AS production

WORKDIR /app

# Install production dependencies for API
COPY api/package*.json ./
RUN npm install --only=production

# Copy the built React app
COPY --from=build /app/build ./build

# Copy production server
COPY production-server.js ./production-server.js

EXPOSE 3001

CMD ["node", "production-server.js"]