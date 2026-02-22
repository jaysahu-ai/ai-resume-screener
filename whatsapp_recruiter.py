from twilio.rest import Client
from openai import OpenAI
import json
from datetime import datetime
from pathlib import Path

class WhatsAppRecruiter:
    def __init__(self, 
                 twilio_account_sid,
                 twilio_auth_token,
                 twilio_whatsapp_number,
                 openai_api_key):
        """
        Initialize WhatsApp Recruitment Bot
        
        Args:
            twilio_account_sid: Your Twilio Account SID
            twilio_auth_token: Your Twilio Auth Token
            twilio_whatsapp_number: Your Twilio WhatsApp number (e.g., +14155238886)
            openai_api_key: OpenAI API key
        """
        self.twilio_client = Client(twilio_account_sid, twilio_auth_token)
        self.twilio_whatsapp = twilio_whatsapp_number
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Store conversations in memory (use database in production)
        self.conversations = {}
        
        # Log all interactions
        self.conversation_log = []
    
    def send_whatsapp_message(self, to_phone, message):
        """
        Send WhatsApp message via Twilio
        
        Args:
            to_phone: Candidate phone number (e.g., +919876543210)
            message: Message text
            
        Returns:
            Message SID if successful, None otherwise
        """
        try:
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=f'whatsapp:{self.twilio_whatsapp}',
                to=f'whatsapp:{to_phone}'
            )
            
            print(f"WhatsApp sent to {to_phone} (SID: {message_obj.sid})")
            
            # Log
            self.conversation_log.append({
                'timestamp': datetime.now().isoformat(),
                'direction': 'outbound',
                'to': to_phone,
                'message': message,
                'sid': message_obj.sid
            })
            
            return message_obj.sid
            
        except Exception as e:
            print(f"WhatsApp send failed: {e}")
            return None
    
    def initiate_conversation(self, candidate_phone, candidate_info, job_info, budget_range):
        """
        Start recruitment conversation with a candidate
        
        Args:
            candidate_phone: Phone number with country code (e.g., +919876543210)
            candidate_info: Dict with candidate details
            job_info: Job description text
            budget_range: Tuple (min_ctc, max_ctc) in LPA
            
        Returns:
            Message SID
        """
        candidate_name = candidate_info.get('name', 'there')
        
        # Generate personalized opening message
        opening_message = f"""Hi {candidate_name}! 👋

I'm reaching out from *{job_info.get('company_name', 'our company')}* regarding an exciting opportunity.

*Role:* {job_info.get('role', 'Senior Position')}
*Location:* {job_info.get('location', 'Bangalore')}

Your profile caught our attention! Would you like to learn more?

Reply with:
✅ *YES* - I'm interested
❌ *NO* - Not right now
ℹ️ *INFO* - Tell me more"""

        # Initialize conversation state
        self.conversations[candidate_phone] = {
            'state': 'initial',
            'candidate_info': candidate_info,
            'job_info': job_info,
            'budget_range': budget_range,
            'messages': [],
            'extracted_data': {},
            'started_at': datetime.now().isoformat()
        }
        
        # Send message
        sid = self.send_whatsapp_message(candidate_phone, opening_message)
        
        # Store in conversation
        self.conversations[candidate_phone]['messages'].append({
            'role': 'assistant',
            'content': opening_message,
            'timestamp': datetime.now().isoformat()
        })
        
        return sid
    
    def process_candidate_reply(self, candidate_phone, reply_text):
        """
        Process candidate's WhatsApp reply and generate response
        
        Args:
            candidate_phone: Candidate phone number
            reply_text: Their reply text
            
        Returns:
            AI-generated response
        """
        # Get conversation context
        conversation = self.conversations.get(candidate_phone)
        
        if not conversation:
            # New conversation (candidate reached out first)
            return "Hi! Thanks for reaching out. Let me connect you with our team."
        
        # Log their message
        conversation['messages'].append({
            'role': 'user',
            'content': reply_text,
            'timestamp': datetime.now().isoformat()
        })
        
        self.conversation_log.append({
            'timestamp': datetime.now().isoformat(),
            'direction': 'inbound',
            'from': candidate_phone,
            'message': reply_text
        })
        
        # Update conversation state based on content
        current_state = conversation['state']
        
        # Generate AI response
        ai_response = self._generate_ai_response(conversation, reply_text)
        
        # Extract information from reply
        extracted = self._extract_information(reply_text, conversation)
        conversation['extracted_data'].update(extracted)
        
        # Update state
        conversation['state'] = self._determine_next_state(conversation, extracted)
        
        # Store AI response
        conversation['messages'].append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Send response
        self.send_whatsapp_message(candidate_phone, ai_response)
        
        return ai_response
    
    def _generate_ai_response(self, conversation, candidate_reply):
        """Generate contextual AI response"""
        
        state = conversation['state']
        job_info = conversation['job_info']
        budget_range = conversation['budget_range']
        candidate_info = conversation['candidate_info']
        extracted = conversation['extracted_data']
        
        # Build system prompt based on state
        system_prompt = f"""You are a friendly recruitment assistant for {job_info.get('company_name', 'our company')}.

JOB DETAILS:
- Role: {job_info.get('role', 'Software Engineer')}
- Location: {job_info.get('location', 'Bangalore')}
- Skills: {job_info.get('skills', 'N/A')}
- Experience: {job_info.get('experience', 'N/A')}
- Budget: {budget_range[0]}-{budget_range[1]} LPA

CANDIDATE INFO:
- Name: {candidate_info.get('name', 'Candidate')}
- Current Score: {candidate_info.get('final_score', 0):.0%} match

CONVERSATION STATE: {state}
ALREADY EXTRACTED: {json.dumps(extracted)}

YOUR GOALS (in order):
1. Build rapport (friendly, professional)
2. Confirm interest
3. Get expected CTC
4. Get availability (notice period)
5. Check if within budget
6. Schedule next steps if good fit

RULES:
- Keep messages SHORT (max 3-4 lines, 50 words)
- Use emojis sparingly (1-2 per message)
- Ask ONE question at a time
- Be conversational, not robotic
- For Indian candidates, use LPA (lakhs per annum) for CTC
- If CTC > budget, politely explain mismatch
- If good fit, offer to schedule call with hiring manager

CURRENT STATE GOALS:
- initial: Confirm if interested
- interested: Ask for expected CTC
- ctc_shared: Ask availability/notice period
- availability_shared: Check budget fit, next steps
- qualified: Schedule interview
- disqualified: Thank politely

Respond naturally to their message:"""

        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add last 5 messages for context
        for msg in conversation['messages'][-5:]:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
        
        # Generate response
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            return "Thanks for your message! Let me get back to you shortly."
    
    def _extract_information(self, message, conversation):
        """Extract structured information from candidate message"""
        
        budget_range = conversation['budget_range']
        
        prompt = f"""Extract information from this candidate message.

Budget range: {budget_range[0]}-{budget_range[1]} LPA

Candidate message: "{message}"

Return ONLY this JSON:
{{
  "interested": true/false/null,
  "expected_ctc": <number in LPA or null>,
  "current_ctc": <number in LPA or null>,
  "availability": "immediate/1week/2weeks/1month/2months/3months/null",
  "notice_period": <number in days or null>,
  "has_questions": true/false,
  "sentiment": "enthusiastic/positive/neutral/hesitant/negative",
  "within_budget": true/false/null
}}

Calculate within_budget by comparing expected_ctc to budget range.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean JSON
            import re
            if result_text.startswith('```'):
                result_text = re.sub(r'^```json?\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)
            
            return json.loads(result_text)
            
        except Exception as e:
            print(f"Error extracting info: {e}")
            return {}
    
    def _determine_next_state(self, conversation, extracted):
        """Determine next conversation state"""
        
        current_state = conversation['state']
        
        # State machine
        if current_state == 'initial':
            if extracted.get('interested') == True:
                return 'interested'
            elif extracted.get('interested') == False:
                return 'not_interested'
        
        elif current_state == 'interested':
            if extracted.get('expected_ctc') is not None:
                return 'ctc_shared'
        
        elif current_state == 'ctc_shared':
            if extracted.get('availability') or extracted.get('notice_period'):
                # Check budget fit
                if extracted.get('within_budget') == True:
                    return 'qualified'
                elif extracted.get('within_budget') == False:
                    return 'over_budget'
                else:
                    return 'availability_shared'
        
        elif current_state == 'availability_shared':
            if extracted.get('within_budget') == True:
                return 'qualified'
            elif extracted.get('within_budget') == False:
                return 'over_budget'
        
        return current_state
    
    def get_qualified_candidates(self):
        """
        Get list of candidates who are interested and within budget
        
        Returns:
            List of qualified candidate dicts
        """
        qualified = []
        
        for phone, conv in self.conversations.items():
            state = conv['state']
            extracted = conv['extracted_data']
            
            if state == 'qualified' or (
                extracted.get('interested') == True and
                extracted.get('within_budget') == True
            ):
                qualified.append({
                    'phone': phone,
                    'name': conv['candidate_info'].get('name'),
                    'expected_ctc': extracted.get('expected_ctc'),
                    'availability': extracted.get('availability'),
                    'notice_period': extracted.get('notice_period'),
                    'state': state,
                    'started_at': conv['started_at']
                })
        
        return qualified
    
    def send_to_hiring_manager(self, qualified_candidates, hiring_manager_phone):
        """Send summary to hiring manager via WhatsApp"""
        
        if not qualified_candidates:
            return "No qualified candidates to send."
        
        summary = f"""*Recruitment Update* 📊

*{len(qualified_candidates)} Qualified Candidates*

"""
        
        for i, candidate in enumerate(qualified_candidates, 1):
            summary += f"""
{i}. *{candidate['name']}*
   💰 Expected: {candidate['expected_ctc']} LPA
   📅 Available: {candidate['availability']}
   📞 {candidate['phone']}
"""
        
        summary += "\n\nReady to schedule interviews! 🎯"
        
        return self.send_whatsapp_message(hiring_manager_phone, summary)
    
    def export_conversations(self, filename="whatsapp_conversations.json"):
        """Export all conversations to JSON"""
        
        export_data = {
            'conversations': self.conversations,
            'log': self.conversation_log,
            'exported_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported to {filename}")
        return filename
    
    def get_conversation_summary(self, candidate_phone):
        """Get summary of conversation with a candidate"""
        
        conv = self.conversations.get(candidate_phone)
        if not conv:
            return "No conversation found"
        
        summary = f"""
{'='*70}
CONVERSATION SUMMARY: {conv['candidate_info'].get('name', 'Unknown')}
{'='*70}

Phone: {candidate_phone}
State: {conv['state']}
Started: {conv['started_at']}
Messages: {len(conv['messages'])}

EXTRACTED DATA:
{json.dumps(conv['extracted_data'], indent=2)}

RECENT MESSAGES:
"""
        for msg in conv['messages'][-5:]:
            role = "BOT" if msg['role'] == 'assistant' else "CANDIDATE"
            summary += f"\n[{role}] {msg['content']}\n"
        
        summary += f"\n{'='*70}\n"
        
        return summary