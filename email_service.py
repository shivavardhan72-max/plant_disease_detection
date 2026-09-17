import os
import re
import resend

def send_prediction_email(to_email, plant_name, disease_name, confidence, ai_recommendation, api_key=None):
    """
    Sends a plant disease diagnostic report via Resend.
    Returns: (success: bool, message: str, delivered_to: str)
    """
    key = api_key or os.environ.get("RESEND_API_KEY")
    if not key:
        msg = "No Resend API key configured. Please add your key in the Profile section or .env file."
        print(f"Warning: {msg}")
        return False, msg, None

    resend.api_key = key
    target_email = to_email or "kjana7037@gmail.com"

    html_content = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 620px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 28px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 24px; letter-spacing: -0.5px;">🌿 FloraScan AI</h1>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 14px;">Agronomic Pathology & Crop Diagnostics</p>
        </div>
        
        <div style="padding: 28px;">
            <p style="font-size: 15px; color: #334155; margin-top: 0;">Hello,</p>
            <p style="font-size: 15px; color: #334155;">Here is the automated pathology diagnosis report generated for your plant specimen:</p>
            
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin: 20px 0;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px 0; color: #64748b; font-size: 14px;">Host Plant:</td>
                        <td style="padding: 8px 0; color: #0f172a; font-weight: 600; font-size: 15px;">{plant_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b; font-size: 14px;">Detected Condition:</td>
                        <td style="padding: 8px 0; color: #0f172a; font-weight: 600; font-size: 15px;">{disease_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; color: #64748b; font-size: 14px;">Model Confidence:</td>
                        <td style="padding: 8px 0; color: #10b981; font-weight: 700; font-size: 16px;">{confidence:.2f}%</td>
                    </tr>
                </table>
            </div>

            <h3 style="color: #0f172a; margin-top: 24px; font-size: 16px; border-bottom: 2px solid #10b981; padding-bottom: 6px; display: inline-block;">
                🌾 AI Agronomist Action Plan & Treatment
            </h3>
            <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 14px 18px; border-radius: 4px; margin: 12px 0 24px 0; color: #166534; font-size: 14px; line-height: 1.6;">
                {ai_recommendation}
            </div>

            <div style="border-top: 1px solid #f1f5f9; padding-top: 18px; color: #94a3b8; font-size: 12px; text-align: center;">
                Generated automatically by FloraScan AI. For persistent foliar symptoms, consult your local agricultural extension service.
            </div>
        </div>
    </div>
    """

    params = {
        "from": "FloraScan AI <onboarding@resend.dev>",
        "to": [target_email],
        "subject": f"🌿 Plant Pathology Report: {plant_name} - {disease_name}",
        "html": html_content
    }

    try:
        email_response = resend.Emails.send(params)
        print(f"Email sent successfully to {target_email}. Response: {email_response}")
        return True, f"Diagnostic report sent successfully to {target_email}!", target_email
    except Exception as e:
        err_msg = str(e)
        print(f"Direct send to {target_email} failed: {err_msg}")
        
        # Check if Resend rejected due to sandbox unverified recipient restriction
        match = re.search(r'\(([^)]+@[^)]+)\)', err_msg)
        if match and "only send testing emails" in err_msg:
            sandbox_email = match.group(1).strip()
            print(f"Attempting fallback delivery to Resend account email: {sandbox_email}")
            try:
                fallback_params = {
                    "from": "FloraScan AI <onboarding@resend.dev>",
                    "to": [sandbox_email],
                    "subject": f"🌿 Plant Pathology Report: {plant_name} - {disease_name} [Sandbox Delivery]",
                    "html": f"""
                    <div style="background: #fffbeb; border: 1px solid #fde68a; color: #92400e; padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; font-family: sans-serif;">
                        <strong>⚠️ Notice from FloraScan AI:</strong><br>
                        This report was requested for <code>{target_email}</code>. Because the active Resend API key is in free sandbox mode (using <code>onboarding@resend.dev</code>), Resend only routes test emails to the account owner (<code>{sandbox_email}</code>). To send directly to <code>{target_email}</code>, verify your custom domain on <a href="https://resend.com/domains">resend.com</a> or use a key from that account.
                    </div>
                    """ + html_content
                }
                fb_response = resend.Emails.send(fallback_params)
                print(f"Fallback email delivered to {sandbox_email}. Response: {fb_response}")
                return True, (
                    f"Email delivered to your Resend account email ({sandbox_email})! "
                    f"Note: To deliver directly to {target_email}, register your domain on resend.com or use an API key registered under {target_email}."
                ), sandbox_email
            except Exception as fb_err:
                print(f"Fallback send failed: {fb_err}")
                return False, f"Failed to send email: {str(fb_err)}", None

        return False, f"Failed to send email: {err_msg}", None

