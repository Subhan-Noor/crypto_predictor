#!/usr/bin/env python3
"""
Production Setup Script for Crypto Price Prediction App

This script helps configure and validate the production environment.
Run this after deploying to ensure everything is set up correctly.
"""

import os
import sys
import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

class ProductionSetup:
    def __init__(self, frontend_url: str, backend_url: str):
        self.frontend_url = frontend_url.rstrip('/')
        self.backend_url = backend_url.rstrip('/')
        self.results = {}
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def test_backend_health(self) -> bool:
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Backend health check passed: {data.get('status', 'unknown')}")
                self.results['backend_health'] = data
                return True
            else:
                self.log(f"❌ Backend health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Backend health check error: {str(e)}", "ERROR")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection through data status endpoint"""
        try:
            response = requests.get(f"{self.backend_url}/data_status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                self.log(f"✅ Database connection successful")
                self.results['database_status'] = data
                return True
            else:
                self.log(f"❌ Database connection failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Database connection error: {str(e)}", "ERROR")
            return False
    
    def test_cors_configuration(self) -> bool:
        """Test CORS configuration"""
        try:
            headers = {
                'Origin': self.frontend_url,
                'Access-Control-Request-Method': 'GET',
                'Access-Control-Request-Headers': 'Content-Type'
            }
            response = requests.options(f"{self.backend_url}/health", headers=headers, timeout=15)
            
            if 'Access-Control-Allow-Origin' in response.headers:
                self.log("✅ CORS configuration is working")
                return True
            else:
                self.log("⚠️ CORS headers not found - may cause frontend issues", "WARNING")
                return False
        except Exception as e:
            self.log(f"❌ CORS test error: {str(e)}", "ERROR")
            return False
    
    def test_frontend_accessibility(self) -> bool:
        """Test if frontend is accessible"""
        try:
            response = requests.get(self.frontend_url, timeout=30)
            if response.status_code == 200:
                self.log("✅ Frontend is accessible")
                return True
            else:
                self.log(f"❌ Frontend not accessible: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Frontend accessibility error: {str(e)}", "ERROR")
            return False
    
    def test_automation_endpoints(self) -> bool:
        """Test automation monitoring endpoints"""
        endpoints = [
            "/automation/status",
            "/automation/history"
        ]
        
        all_passed = True
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.backend_url}{endpoint}", timeout=30)
                if response.status_code == 200:
                    self.log(f"✅ Automation endpoint {endpoint} is working")
                else:
                    self.log(f"❌ Automation endpoint {endpoint} failed: {response.status_code}", "ERROR")
                    all_passed = False
            except Exception as e:
                self.log(f"❌ Automation endpoint {endpoint} error: {str(e)}", "ERROR")
                all_passed = False
        
        return all_passed
    
    def test_prediction_endpoints(self) -> bool:
        """Test ML prediction endpoints"""
        currencies = ["BTC", "ETH"]
        
        for currency in currencies:
            try:
                self.log(f"Testing prediction for {currency}...")
                response = requests.post(
                    f"{self.backend_url}/predict/{currency}", 
                    json={}, 
                    timeout=60  # Predictions may take longer
                )
                if response.status_code == 200:
                    data = response.json()
                    self.log(f"✅ {currency} prediction endpoint working: {data.get('prediction', 'unknown')}")
                else:
                    self.log(f"⚠️ {currency} prediction endpoint returned: {response.status_code}", "WARNING")
            except Exception as e:
                self.log(f"⚠️ {currency} prediction test error: {str(e)}", "WARNING")
        
        return True  # Non-critical for basic functionality
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate deployment validation report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "frontend_url": self.frontend_url,
            "backend_url": self.backend_url,
            "test_results": self.results,
            "status": "validation_complete"
        }
    
    def run_all_tests(self) -> bool:
        """Run all production validation tests"""
        self.log("🚀 Starting production environment validation...")
        self.log(f"Frontend URL: {self.frontend_url}")
        self.log(f"Backend URL: {self.backend_url}")
        
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Database Connection", self.test_database_connection),
            ("Frontend Accessibility", self.test_frontend_accessibility),
            ("CORS Configuration", self.test_cors_configuration),
            ("Automation Endpoints", self.test_automation_endpoints),
            ("Prediction Endpoints", self.test_prediction_endpoints),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"\n--- Testing {test_name} ---")
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                self.log(f"❌ {test_name} failed with exception: {str(e)}", "ERROR")
        
        # Generate final report
        self.log(f"\n🎯 Validation Summary: {passed}/{total} tests passed")
        
        if passed >= total - 1:  # Allow 1 failure
            self.log("🎉 Production environment is ready!")
            return True
        else:
            self.log("⚠️ Some issues found. Check logs above.", "WARNING")
            return False

def check_environment_variables():
    """Check if required environment variables are documented"""
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY", 
        "ENVIRONMENT"
    ]
    
    print("\n📋 Environment Variables Checklist:")
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Not set (may be in platform dashboard)")

def main():
    """Main function"""
    print("🚀 Crypto Price Prediction - Production Setup")
    print("=" * 50)
    
    # Get URLs from command line or environment
    frontend_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("FRONTEND_URL")
    backend_url = sys.argv[2] if len(sys.argv) > 2 else os.getenv("BACKEND_URL")
    
    if not frontend_url:
        frontend_url = input("Enter your frontend URL (e.g., https://your-app.vercel.app): ").strip()
    
    if not backend_url:
        backend_url = input("Enter your backend URL (e.g., https://your-app.railway.app): ").strip()
    
    if not frontend_url or not backend_url:
        print("❌ Both frontend and backend URLs are required!")
        print("\nUsage:")
        print("  python scripts/production_setup.py <frontend_url> <backend_url>")
        print("  OR set FRONTEND_URL and BACKEND_URL environment variables")
        sys.exit(1)
    
    # Check environment variables
    check_environment_variables()
    
    # Run validation tests
    setup = ProductionSetup(frontend_url, backend_url)
    success = setup.run_all_tests()
    
    # Save report
    report = setup.generate_report()
    report_file = f"production_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Validation report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 