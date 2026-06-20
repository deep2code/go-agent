#!/bin/bash
# ============================================================================
# go-agent 编译 & 安装脚本
#
# 用法:
#   ./build.sh           编译并安装到 $GOPATH/bin
#   ./build.sh -l        仅编译（输出到当前目录）
#   ./build.sh -c        清理编译产物
#   ./build.sh -h        显示帮助
#
# 安装后可在任意目录直接执行: go-agent
# ============================================================================

set -euo pipefail

# ── 颜色 ──────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── 配置 ──────────────────────────────────────────
BINARY_NAME="go-agent"
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SOURCE_DIR/build"
GOPATH="${GOPATH:-$(go env GOPATH)}"
GOBIN="${GOBIN:-$GOPATH/bin}"

# ── 打印函数 ──────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 帮助 ──────────────────────────────────────────
usage() {
    cat << EOF
${CYAN}go-agent 编译脚本${NC}

用法: $0 [选项]

选项:
  (无参数)    编译并安装到 \$GOPATH/bin ($GOPATH/bin)
  -l          仅本地编译（输出到 $BUILD_DIR/）
  -c          清理编译产物
  -o <path>   指定输出路径
  -v          显示版本信息后编译
  -h          显示本帮助

环境变量:
  GOPATH  Go 工作目录 (当前: $GOPATH)
  GOBIN   二进制安装目录 (当前: $GOBIN)

示例:
  $0              # 编译并全局安装
  $0 -l           # 仅编译到本地 build/
  $0 -o ./myagent # 编译到指定路径
EOF
}

# ── 版本信息 ──────────────────────────────────────
show_version() {
    # 尝试从 git 获取版本，否则用默认值
    if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null 2>&1; then
        GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
    else
        GIT_COMMIT="unknown"
        GIT_TAG=""
    fi
    BUILD_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    GO_VERSION=$(go version | awk '{print $3}')

    echo "  版本:     ${GIT_TAG:-dev} (commit ${GIT_COMMIT})"
    echo "  构建时间: ${BUILD_TIME}"
    echo "  Go 版本:  ${GO_VERSION}"
    echo "  目标目录: ${GOBIN}"
    echo ""
}

# ── 编译 ──────────────────────────────────────────
do_build() {
    local output="$1"
    local ldflags=""

    # 注入版本信息
    if command -v git &>/dev/null && git rev-parse --git-dir &>/dev/null 2>&1; then
        GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        BUILD_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
        ldflags="-s -w -X main.gitCommit=${GIT_COMMIT} -X main.buildTime=${BUILD_TIME}"
    fi

    info "编译 $BINARY_NAME ..."
    info "  源目录: $SOURCE_DIR"
    info "  输出:   $output"

    cd "$SOURCE_DIR"

    if [ -n "$ldflags" ]; then
        go build -ldflags "$ldflags" -o "$output" .
    else
        go build -o "$output" .
    fi

    if [ -x "$output" ]; then
        local size
        size=$(ls -lh "$output" | awk '{print $5}')
        success "编译成功: $output ($size)"
    else
        error "编译失败"
        exit 1
    fi
}

# ── 安装到 GOBIN ─────────────────────────────────
do_install() {
    local binary_path="$BUILD_DIR/$BINARY_NAME"

    # 先编译
    do_build "$binary_path"

    # 确保目标目录存在
    mkdir -p "$GOBIN"

    # 复制
    cp "$binary_path" "$GOBIN/$BINARY_NAME"
    chmod +x "$GOBIN/$BINARY_NAME"

    success "已安装到: $GOBIN/$BINARY_NAME"

    # 检查是否在 PATH 中
    if [[ ":$PATH:" != *":$GOBIN:"* ]]; then
        warn "注意: $GOBIN 不在 PATH 中"
        echo "  请添加以下行到 ~/.zshrc 或 ~/.bashrc:"
        echo ""
        echo "    export PATH=\"\$GOPATH/bin:\$PATH\""
        echo ""
    else
        success "已就绪，可在任意目录执行: ${CYAN}go-agent${NC}"
    fi
}

# ── 清理 ──────────────────────────────────────────
do_clean() {
    info "清理编译产物..."
    rm -rf "$BUILD_DIR"
    rm -f "$SOURCE_DIR/$BINARY_NAME"
    success "清理完成"
}

# ── 主流程 ──────────────────────────────────────────

# 解析参数
MODE="install"        # install | local | clean
CUSTOM_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -l|--local)
            MODE="local"
            shift
            ;;
        -c|--clean)
            MODE="clean"
            shift
            ;;
        -v|--version)
            show_version
            shift
            ;;
        -o|--output)
            CUSTOM_OUTPUT="$2"
            MODE="custom"
            shift 2
            ;;
        *)
            error "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  go-agent 编译脚本${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

case "$MODE" in
    install)
        show_version
        do_install
        ;;
    local)
        show_version
        mkdir -p "$BUILD_DIR"
        do_build "$BUILD_DIR/$BINARY_NAME"
        ;;
    clean)
        do_clean
        ;;
    custom)
        do_build "$CUSTOM_OUTPUT"
        ;;
esac

echo ""
