# Docker Setup for Next.js App with ONNX and WASM

This guide explains how to run the Next.js application with ONNX model and Rust WASM package in Docker containers.

## Prerequisites

- Docker and Docker Compose installed on your system
- The ONNX model file (`tiny_mel_classifier.onnx`) in the root directory

## Project Structure

```
.
├── app/                          # Next.js application
│   ├── components/
│   ├── lib/
│   ├── pages/
│   ├── public/
│   │   └── tiny_mel_classifier.onnx  # ONNX model (mounted from root)
│   ├── Dockerfile                # Production Dockerfile
│   ├── Dockerfile.dev           # Development Dockerfile
│   ├── next.config.js           # Next.js config with WASM support
│   └── package.json             # Dependencies including ONNX and WASM
├── docker-compose.yml           # Production setup
├── docker-compose.dev.yml       # Development setup
└── tiny_mel_classifier.onnx     # ONNX model file
```

## Quick Start

### Production Build

1. **Build and run the production container:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   Open your browser and navigate to `http://localhost:3000`

### Development Mode

1. **Run in development mode with hot reloading:**
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

2. **Access the development server:**
   Open your browser and navigate to `http://localhost:3000`

## Manual Docker Commands

### Production Build

```bash
# Build the production image
cd app
docker build -t nextjs-onnx-app .

# Run the container
docker run -p 3000:3000 \
  -v $(pwd)/../tiny_mel_classifier.onnx:/app/public/tiny_mel_classifier.onnx:ro \
  nextjs-onnx-app
```

### Development Build

```bash
# Build the development image
cd app
docker build -f Dockerfile.dev -t nextjs-onnx-app-dev .

# Run the development container
docker run -p 3000:3000 \
  -v $(pwd):/app \
  -v /app/node_modules \
  -v /app/.next \
  -v $(pwd)/../tiny_mel_classifier.onnx:/app/public/tiny_mel_classifier.onnx:ro \
  nextjs-onnx-app-dev
```

## Key Features

### WASM Support
- Configured in `next.config.js` with `asyncWebAssembly: true`
- Proper webpack configuration for WASM modules
- Cross-origin headers for SharedArrayBuffer support

### ONNX Runtime Web
- Included in dependencies for browser-based ML inference
- Model loaded from `/public/tiny_mel_classifier.onnx`
- Optimized for client-side execution

### Multi-stage Build
- Optimized production image using Next.js standalone output
- Separate development container for faster iteration
- Minimal runtime dependencies

## Environment Variables

You can customize the behavior using environment variables:

```bash
# Disable Next.js telemetry
NEXT_TELEMETRY_DISABLED=1

# Set custom port (default: 3000)
PORT=3000

# Set hostname
HOSTNAME=0.0.0.0
```

## Troubleshooting

### WASM Loading Issues
If you encounter WASM loading problems:
1. Ensure the rust-melspec-wasm package is properly installed
2. Check browser console for CORS errors
3. Verify the Cross-Origin headers are set correctly

### ONNX Model Loading
If the ONNX model fails to load:
1. Verify the model file exists in the public directory
2. Check the model path in the application code
3. Ensure the model file is properly mounted in Docker

### Performance Issues
For better performance:
1. Use the production build (`docker-compose.yml`)
2. Ensure adequate memory allocation for Docker
3. Consider using a CDN for the ONNX model in production

## Stopping the Application

```bash
# Stop and remove containers
docker-compose down

# Stop development containers
docker-compose -f docker-compose.dev.yml down

# Remove images (optional)
docker-compose down --rmi all
```

## Additional Notes

- The application requires modern browser support for WebAssembly and SharedArrayBuffer
- ONNX model inference runs entirely in the browser
- The Rust WASM package provides optimized audio processing capabilities
- Hot reloading is available in development mode for faster iteration 