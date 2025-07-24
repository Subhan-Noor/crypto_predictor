#!/usr/bin/env python3
"""
Test script for current prices endpoint
"""

import asyncio
import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.binance_service import BinancePriceFetcher

async def test_binance_service():
    """Test the Binance service directly"""
    print("🧪 Testing Binance Service...")
    
    try:
        binance_service = BinancePriceFetcher()
        
        # Test BTC price
        print("📊 Fetching BTC price...")
        btc_price = await binance_service.get_current_price("BTCUSDT")
        print(f"✅ BTC Price: {btc_price}")
        
        # Test ETH price
        print("📊 Fetching ETH price...")
        eth_price = await binance_service.get_current_price("ETHUSDT")
        print(f"✅ ETH Price: {eth_price}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Binance service: {e}")
        return False

async def test_current_prices_endpoint():
    """Test the current prices endpoint"""
    print("\n🧪 Testing Current Prices Endpoint...")
    
    try:
        from app.enhanced_main import get_enhanced_current_prices
        
        # Test the endpoint
        result = await get_enhanced_current_prices()
        print(f"✅ Current Prices Result: {result}")
        
        # Check structure - should have BTC and ETH directly
        if "BTC" in result and "ETH" in result:
            btc_data = result["BTC"]
            eth_data = result["ETH"]
            
            # Check required fields
            required_fields = ["currency", "price", "change_24h", "change_percentage_24h", "volume_24h", "last_updated"]
            
            btc_ok = all(field in btc_data for field in required_fields)
            eth_ok = all(field in eth_data for field in required_fields)
            
            if btc_ok and eth_ok:
                print("✅ Response structure is correct")
                print(f"📊 BTC Price: ${btc_data['price']:,.2f}")
                print(f"📊 ETH Price: ${eth_data['price']:,.2f}")
                return True
            else:
                print("❌ Missing required fields in response")
                return False
        else:
            print("❌ Response missing BTC or ETH data")
            return False
            
    except Exception as e:
        print(f"❌ Error testing current prices endpoint: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Current Prices Test Script")
    print("=" * 40)
    
    # Test 1: Binance Service
    binance_ok = await test_binance_service()
    
    # Test 2: Current Prices Endpoint
    endpoint_ok = await test_current_prices_endpoint()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"Binance Service: {'✅ PASS' if binance_ok else '❌ FAIL'}")
    print(f"Current Prices Endpoint: {'✅ PASS' if endpoint_ok else '❌ FAIL'}")
    
    if binance_ok and endpoint_ok:
        print("\n🎉 All tests passed! Current prices should work.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 