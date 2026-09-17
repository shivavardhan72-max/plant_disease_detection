import io
import unittest
from PIL import Image
import numpy as np

from app import app
import model


class PlantDiseaseDetectionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()
        
        # Ensure test user exists and login
        import db
        user = db.get_user_by_email("testfarmer@gmail.com")
        if not user:
            user = db.create_user("testfarmer@gmail.com", "password123", "Test Farmer")
            db.update_user_profile(user.id, notification_email="kjana7037@gmail.com")
        cls.client.post("/login", data={"email": "testfarmer@gmail.com", "password": "password123"})

    def test_model_direct_inference(self):
        """Test model.py prediction pipeline with a generated PIL image."""
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        result = model.predict_disease(img, top_k=5)
        
        self.assertTrue(result["success"])
        self.assertIn("predicted_class", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)
        self.assertEqual(len(result["top_predictions"]), 5)
        print(f"[OK] Direct inference passed: {result['plant']} - {result['disease']} ({result['confidence']}%)")

    def test_dashboard_route(self):
        """Test that GET /dashboard returns the HTML interface."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FloraScan", response.data)
        self.assertIn(b"Diagnose Specimen", response.data)
        print("[OK] Dashboard route test passed.")

    def test_predict_multipart_file_upload(self):
        """Test POST /predict with multipart file upload."""
        img = Image.new("RGB", (224, 224), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        data = {
            "file": (buf, "test_leaf.png")
        }
        response = self.client.post(
            "/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("predicted_class", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("top_predictions", json_data)
        print(f"[OK] File upload prediction passed: {json_data['predicted_class']} ({json_data['confidence']}%)")

    def test_predict_sample_preset(self):
        """Test POST /predict with preset sample name."""
        response = self.client.post(
            "/predict",
            json={"sample": "sample_healthy.jpg"}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertEqual(json_data["plant"], "Tomato")
        self.assertEqual(json_data["disease"], "Healthy & Vigorous")
        self.assertGreater(json_data["confidence"], 90.0)
        print(f"[OK] Preset sample prediction passed: {json_data['predicted_class']} ({json_data['confidence']}%)")

    def test_send_email_report_endpoint(self):
        """Test POST /api/send-email-report with latest prediction."""
        response = self.client.post(
            "/api/send-email-report",
            json={}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data["success"])
        self.assertIn("message", json_data)
        print(f"[OK] Send email report passed: {json_data['message']}")


if __name__ == "__main__":
    unittest.main()
