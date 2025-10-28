# Build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies for building)
RUN npm install

# Copy source code
COPY . .

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