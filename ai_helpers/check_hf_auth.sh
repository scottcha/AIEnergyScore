#!/bin/bash
# Helper script to check HuggingFace authentication status
# Usage: ./check_hf_auth.sh

set -e

echo "============================================"
echo "HuggingFace Authentication Status Check"
echo "============================================"
echo ""

# Check for HF_TOKEN environment variable
if [ -n "${HF_TOKEN}" ]; then
    echo "✓ HF_TOKEN environment variable is set"
    TOKEN_LENGTH=${#HF_TOKEN}
    echo "  Token length: ${TOKEN_LENGTH} characters"
    echo "  Token prefix: ${HF_TOKEN:0:7}..."
    HF_AUTH_FOUND=true
else
    echo "✗ HF_TOKEN environment variable is not set"
    HF_AUTH_FOUND=false
fi

echo ""

# Check for token file
HF_TOKEN_FILE="${HOME}/.huggingface/token"
if [ -f "${HF_TOKEN_FILE}" ]; then
    echo "✓ HuggingFace token file found at: ${HF_TOKEN_FILE}"
    # Read token and show prefix (safely)
    TOKEN_CONTENT=$(cat "${HF_TOKEN_FILE}" | tr -d '\n' | tr -d ' ')
    TOKEN_LENGTH=${#TOKEN_CONTENT}
    echo "  Token length: ${TOKEN_LENGTH} characters"
    echo "  Token prefix: ${TOKEN_CONTENT:0:7}..."
    HF_AUTH_FOUND=true
else
    echo "✗ HuggingFace token file not found at: ${HF_TOKEN_FILE}"
fi

echo ""

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    echo "✓ huggingface-cli is installed"
    echo ""
    echo "Checking authentication status with huggingface-cli..."
    if huggingface-cli whoami &> /dev/null; then
        echo "✓ Successfully authenticated with HuggingFace Hub"
        huggingface-cli whoami
        HF_AUTH_FOUND=true
    else
        echo "✗ Not authenticated with HuggingFace Hub"
        echo "  Run: huggingface-cli login"
    fi
else
    echo "✗ huggingface-cli is not installed"
    echo "  Install with: pip install -U huggingface_hub"
fi

echo ""
echo "============================================"
if [ "${HF_AUTH_FOUND}" = true ]; then
    echo "✓ Authentication configured - you can access gated models"
else
    echo "✗ No authentication found"
    echo ""
    echo "To authenticate, choose one of these methods:"
    echo ""
    echo "Method 1 (Recommended): Login via CLI"
    echo "  1. Install: pip install -U huggingface_hub"
    echo "  2. Login: huggingface-cli login"
    echo "  3. Paste your token from https://huggingface.co/settings/tokens"
    echo ""
    echo "Method 2: Set environment variable"
    echo "  export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "After authenticating, verify access to gated models:"
    echo "  huggingface-cli whoami"
fi
echo "============================================"
