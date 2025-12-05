# Script de instalação para o projeto YOLO Training
# Instala PyTorch com suporte a CUDA e demais dependências

Write-Host "🚀 Iniciando instalação das dependências..." -ForegroundColor Cyan
Write-Host ""

# Verificar se está em um ambiente virtual
if (-not $env:VIRTUAL_ENV) {
    Write-Host "⚠️  AVISO: Nenhum ambiente virtual detectado!" -ForegroundColor Yellow
    Write-Host "   Recomenda-se ativar o venv antes: .\venv\Scripts\activate" -ForegroundColor Yellow
    $continue = Read-Host "Deseja continuar mesmo assim? (s/N)"
    if ($continue -ne "s" -and $continue -ne "S") {
        Write-Host "❌ Instalação cancelada." -ForegroundColor Red
        exit 1
    }
    Write-Host ""
}

# 1. Atualizar pip, setuptools e wheel
Write-Host "📦 Atualizando pip, setuptools e wheel..." -ForegroundColor Green
py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erro ao atualizar pip" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 2. Instalar PyTorch com CUDA
Write-Host "🔥 Instalando PyTorch com suporte a CUDA 12.4..." -ForegroundColor Green
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erro ao instalar PyTorch" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 3. Instalar demais dependências
Write-Host "📚 Instalando demais dependências..." -ForegroundColor Green
py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erro ao instalar dependências" -ForegroundColor Red
    exit 1
}
Write-Host ""

# 4. Verificar instalação
Write-Host "✅ Verificando instalação..." -ForegroundColor Green
py -c "import torch; print(f'PyTorch versão: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'CUDA versão: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA: Não disponível')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Aviso: Erro ao verificar instalação" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "🎉 Instalação concluída com sucesso!" -ForegroundColor Green
}
