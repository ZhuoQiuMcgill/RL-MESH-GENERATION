#!/usr/bin/env python3
"""
测试HistoryManager性能优化效果的脚本
"""

import time
import logging
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rl.training.history_manager import HistoryManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_performance():
    """
    测试HistoryManager的性能优化效果
    """
    print("=" * 60)
    print("HistoryManager 性能测试")
    print("=" * 60)
    
    # 创建HistoryManager实例
    hm = HistoryManager()
    
    # 获取可用的训练ID列表
    training_ids = hm.list_training_id()
    if not training_ids:
        print("❌ 没有找到训练历史数据，无法进行测试")
        return
    
    # 选择第一个训练ID进行测试
    test_training_id = training_ids[0]
    print(f"📊 使用训练ID进行测试: {test_training_id}")
    
    # 测试聚焦操作
    print("\n1. 测试focus_on操作...")
    start_time = time.time()
    hm.focus_on(test_training_id)
    focus_time = time.time() - start_time
    print(f"   ✅ focus_on耗时: {focus_time:.4f}秒")
    
    # 获取缓存状态
    cache_status = hm.get_cache_status()
    print(f"   📋 数据大小: {cache_status['data_size']} episodes")
    print(f"   🎯 缓存状态: {'有效' if cache_status['cache_valid'] else '无效'}")
    
    if cache_status['data_size'] == 0:
        print("❌ 没有episode数据，无法进行详细测试")
        return
    
    # 测试多次读取同一个episode的性能
    print("\n2. 测试连续读取相同episode的性能...")
    episode_index = 0
    num_reads = 50
    
    start_time = time.time()
    for i in range(num_reads):
        episode_data = hm.get_episode_data(episode_index)
    consecutive_time = time.time() - start_time
    
    print(f"   ✅ 连续读取{num_reads}次相同episode耗时: {consecutive_time:.4f}秒")
    print(f"   📈 平均每次读取耗时: {consecutive_time/num_reads:.6f}秒")
    
    # 测试读取不同episode的性能
    print("\n3. 测试读取不同episode的性能...")
    max_episodes = min(20, cache_status['data_size'])
    
    start_time = time.time()
    for i in range(max_episodes):
        episode_data = hm.get_episode_data(i)
    different_episodes_time = time.time() - start_time
    
    print(f"   ✅ 读取{max_episodes}个不同episode耗时: {different_episodes_time:.4f}秒")
    print(f"   📈 平均每次读取耗时: {different_episodes_time/max_episodes:.6f}秒")
    
    # 测试缓存有效性检查
    print("\n4. 测试缓存有效性...")
    is_valid_before = hm.is_cache_valid()
    print(f"   🔍 检查前缓存状态: {'有效' if is_valid_before else '无效'}")
    
    # 测试强制刷新
    print("\n5. 测试强制刷新功能...")
    start_time = time.time()
    hm.force_refresh()
    refresh_time = time.time() - start_time
    print(f"   🔄 强制刷新耗时: {refresh_time:.4f}秒")
    
    # 测试统计信息获取
    print("\n6. 测试统计信息获取...")
    start_time = time.time()
    stats = hm.get_statistics()
    stats_time = time.time() - start_time
    print(f"   📊 获取统计信息耗时: {stats_time:.4f}秒")
    print(f"   📋 统计信息: {stats['non_zero_episodes']} episodes, 平均奖励: {stats['avg_reward']:.2f}")
    
    # 性能总结
    print("\n" + "=" * 60)
    print("📊 性能测试总结")
    print("=" * 60)
    print(f"初始加载耗时:     {focus_time:.4f}秒")
    print(f"连续读取性能:     {consecutive_time/num_reads:.6f}秒/次")
    print(f"不同episode读取:  {different_episodes_time/max_episodes:.6f}秒/次")
    print(f"强制刷新耗时:     {refresh_time:.4f}秒")
    print(f"统计信息获取:     {stats_time:.4f}秒")
    
    # 计算性能提升估算
    if consecutive_time > 0:
        old_estimated_time = num_reads * focus_time  # 假设每次都重新读取文件
        improvement_ratio = old_estimated_time / consecutive_time
        print(f"\n🚀 估算性能提升: {improvement_ratio:.1f}x (基于连续读取测试)")
    
    print("\n✅ 性能测试完成!")

if __name__ == "__main__":
    try:
        test_performance()
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()