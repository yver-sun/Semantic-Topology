"""硬件检测和配置建议脚本"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_pytorch():
    """检查PyTorch和CUDA支持"""
    try:
        import torch
        logger.info(f"✅ PyTorch版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA版本: {torch.version.cuda}")
            logger.info("   建议：可以使用GPU加速")
        else:
            logger.info("ℹ️  CUDA不可用（这是正常的，AMD GPU需要ROCm）")
            logger.info("   建议：使用CPU模式（Ryzen 7 9700X性能很强）")
        
        return cuda_available
    except ImportError:
        logger.error("❌ PyTorch未安装")
        return False


def check_cpu():
    """检查CPU信息"""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        logger.info(f"✅ CPU核心数: {cpu_count} (逻辑核心)")
        if cpu_freq:
            logger.info(f"   CPU频率: {cpu_freq.current:.0f} MHz")
        
        # 检查CPU型号（Windows）
        try:
            import platform
            if platform.system() == "Windows":
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'name'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cpu_name = lines[1].strip()
                        logger.info(f"   CPU型号: {cpu_name}")
        except:
            pass
        
        return cpu_count
    except ImportError:
        logger.warning("⚠️  psutil未安装，无法检测CPU详细信息")
        logger.info("   安装: pip install psutil")
        return None


def check_memory():
    """检查内存"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        logger.info(f"✅ 总内存: {total_gb:.1f} GB")
        logger.info(f"   可用内存: {available_gb:.1f} GB")
        
        if total_gb >= 16:
            logger.info("   ✅ 内存充足，可以增加批处理大小")
            return True
        elif total_gb >= 8:
            logger.info("   ⚠️  内存一般，建议使用默认配置")
            return False
        else:
            logger.warning("   ⚠️  内存较少，建议减小批处理大小")
            return False
    except ImportError:
        logger.warning("⚠️  psutil未安装，无法检测内存")
        return None


def check_disk_space():
    """检查磁盘空间"""
    try:
        import psutil
        disk = psutil.disk_usage('.')
        total_gb = disk.total / (1024**3)
        free_gb = disk.free / (1024**3)
        
        logger.info(f"✅ 磁盘空间:")
        logger.info(f"   总空间: {total_gb:.1f} GB")
        logger.info(f"   可用空间: {free_gb:.1f} GB")
        
        if free_gb < 10:
            logger.warning("   ⚠️  磁盘空间较少，注意清理")
        
        return free_gb
    except ImportError:
        return None


def print_recommendations():
    """打印配置建议"""
    logger.info("\n" + "="*60)
    logger.info("配置建议")
    logger.info("="*60)
    
    logger.info("\n📝 根据您的硬件配置（Ryzen 7 9700X + 32GB内存）：")
    logger.info("\n1. GPU配置：")
    logger.info("   - 您的RX590是AMD显卡，不支持PyTorch的CUDA")
    logger.info("   - 建议：在config/default_config.yaml中设置 device: 'cpu'")
    logger.info("   - Ryzen 7 9700X的CPU性能很强，足够快速处理")
    
    logger.info("\n2. 批处理优化：")
    logger.info("   - 32GB内存允许更大的批处理")
    logger.info("   - 建议：将 batch_size 从 32 增加到 48-64")
    
    logger.info("\n3. TDA分析优化：")
    logger.info("   - 可以增加地标数量以提升精度")
    logger.info("   - 建议：将 n_landmarks 从 512 增加到 768-1024")
    
    logger.info("\n4. 环境变量优化：")
    logger.info("   - 设置 OMP_NUM_THREADS=16（利用所有核心）")
    logger.info("   - 设置 SPACY_NUM_JOBS=16（Spacy并行处理）")
    
    logger.info("\n📖 详细优化指南：docs/HARDWARE_OPTIMIZATION.md")


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("硬件检测和配置建议")
    logger.info("="*60)
    logger.info("")
    
    # 检查各个组件
    logger.info("检查PyTorch...")
    cuda_available = check_pytorch()
    logger.info("")
    
    logger.info("检查CPU...")
    cpu_count = check_cpu()
    logger.info("")
    
    logger.info("检查内存...")
    memory_ok = check_memory()
    logger.info("")
    
    logger.info("检查磁盘...")
    disk_free = check_disk_space()
    logger.info("")
    
    # 打印建议
    print_recommendations()


if __name__ == '__main__':
    main()

