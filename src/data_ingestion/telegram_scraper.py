import asyncio
import json
import base64
import os
from telethon import TelegramClient, errors
from telethon.sessions import StringSession
import pandas as pd
from pathlib import Path
import logging
import sys
import io
import time
from datetime import datetime
from typing import List, Dict, Optional, Union
import hashlib
import signal
import backoff
import pytz
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import zipfile
from io import BytesIO

# Enhanced logging configuration with single-line progress
class SingleLineProgressHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.last_message = ""
    
    def emit(self, record):
        msg = self.format(record)
        if msg.startswith("PROGRESS:"):
            sys.stdout.write('\r' + msg[9:].ljust(len(self.last_message)))
            sys.stdout.flush()
            self.last_message = msg[9:]
        else:
            if self.last_message:
                sys.stdout.write('\n')
                self.last_message = ""
            sys.stdout.write(msg + '\n')
            sys.stdout.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("telethon_scraper.log", encoding='utf-8', mode='a'),
        SingleLineProgressHandler()
    ]
)
logger = logging.getLogger(__name__)

# Signal handling
shutdown_event = asyncio.Event()

def handle_signal(signum, frame):
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# Configuration with validation
class Config:
    def __init__(self):
        self.api_id = 22662015
        self.api_hash = '3fb4287ba31cc3295b6658a23c56a435'
        self.phone_number = '+251911914562'
        self.session_string = self.get_valid_session()
        self.channels = [
            'telegram',  # Test with public channel first
            'ZemenExpress',
            'nevacomputer',
            'Shageronlinestore'
        ]
        self.max_retries = 5
        self.request_timeout = 300
        self.rate_limit_delay = 1.0
        self.max_concurrent_tasks = 5
        self.max_concurrent_downloads = 5
        self.output_dir = Path("data/raw")
        self.media_dir = Path("data/media")
        self.timezone = pytz.timezone('Africa/Addis_Ababa')
        self.create_directories()

    def get_valid_session(self):
        """Get or create a valid session string"""
        # Try environment variable first
        env_session = os.getenv('TG_SESSION_STRING')
        if env_session and self.validate_session_string(env_session):
            logger.info("Using session from environment variable")
            return env_session
        
        # Try to load from file
        session_file = Path('session.txt')
        if session_file.exists():
            with open(session_file, 'r') as f:
                file_session = f.read().strip()
                if self.validate_session_string(file_session):
                    logger.info("Using session from session.txt")
                    return file_session
        
        # No valid session found
        logger.warning("No valid session found, will create new one")
        return None

    @staticmethod
    def validate_session_string(session_str: str) -> bool:
        """Validate the session string format"""
        if not session_str:
            return False
        try:
            # Check if it's a valid base64 string
            if len(session_str) % 4 != 0:
                session_str += '=' * (4 - len(session_str) % 4)
            decoded = base64.urlsafe_b64decode(session_str)
            return len(decoded) > 20  # Basic length check
        except Exception as e:
            logger.debug(f"Session validation failed: {str(e)}")
            return False

    def create_directories(self):
        """Create output directories with permission check"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.media_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = self.output_dir / 'permission_test.txt'
            test_file.write_text('test')
            test_file.unlink()
            logger.info("Verified directory permissions")
        except Exception as e:
            logger.error(f"Directory creation failed: {str(e)}")
            raise

config = Config()

# Helper functions
def generate_file_hash(content: Union[str, bytes]) -> str:
    return hashlib.sha256(content.encode('utf-8') if isinstance(content, str) else content).hexdigest()

def safe_encode(text: str) -> str:
    return text.encode('utf-8', errors='replace').decode('utf-8')

def format_timestamp(dt: datetime) -> str:
    return dt.astimezone(config.timezone).isoformat()

# Enhanced Telegram scraper
class TelegramScraper:
    def __init__(self, client: TelegramClient):
        self.client = client
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.scraped_count = 0
        self.failed_count = 0
        self.all_messages = []  # Store all messages for combined CSV
        self.current_download = None

    async def download_media(self, message, channel_name: str) -> Optional[Dict]:
        if not message.media:
            return None

        try:
            # Skip if it's not a photo (we only want images)
            if not hasattr(message.media, 'photo'):
                return None

            safe_channel = channel_name.replace('@', '').replace('/', '_')
            timestamp = int(message.date.timestamp())
            
            # Only allow image extensions
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
            file_ext = '.jpg'  # Default extension for photos
            
            filename = f"{safe_channel}_{message.id}_{timestamp}{file_ext}"
            media_path = config.media_dir / filename

            # Progress callback
            def progress_callback(current, total):
                self.current_download = f"Downloading {filename}: {current/1024:.1f}KB/{total/1024:.1f}KB ({current/total:.1%})"
                logger.info(f"PROGRESS:{self.current_download}")

            # Download with proper parameters
            try:
                await asyncio.wait_for(
                    self.client.download_media(
                        message.media,
                        file=media_path,
                        progress_callback=progress_callback
                    ),
                    timeout=config.request_timeout
                )
                # Clear progress line after download completes
                logger.info("PROGRESS:" + " " * len(self.current_download))
                logger.info(f"PROGRESS:Download complete: {filename}")

                return {
                    'media_path': str(media_path),
                    'media_type': file_ext[1:],
                    'media_size': media_path.stat().st_size
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading {filename}")
                return None
            except Exception as e:
                logger.warning(f"Download failed for {filename}: {str(e)}")
                return None
            
        except Exception as e:
            logger.warning(f"Media processing failed: {str(e)}")
            return None

    async def create_zip_archive(self, channel_name: str):
        """Create ZIP archive of all downloaded images for a channel"""
        safe_name = channel_name.replace('@', '').replace('/', '_')
        zip_filename = config.media_dir / f"{safe_name}_images.zip"
        
        # Create in-memory zip first
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            for media_file in config.media_dir.glob(f"{safe_name}_*"):
                if media_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    zf.write(media_file, media_file.name)
                    logger.debug(f"Added {media_file.name} to ZIP")
        
        # Write to disk
        with open(zip_filename, 'wb') as f:
            f.write(mem_zip.getvalue())
        
        logger.info(f"Created images ZIP archive: {zip_filename}")
        return str(zip_filename)

    @backoff.on_exception(
        backoff.expo,
        (errors.FloodWaitError, errors.ServerError, asyncio.TimeoutError),
        max_tries=config.max_retries
    )
    async def scrape_channel(self, channel_name: str) -> List[Dict]:
        if shutdown_event.is_set():
            return []

        try:
            if not channel_name.startswith('@'):
                channel_name = f'@{channel_name}'
            
            logger.info(f"Starting scrape for: {channel_name}")
            
            entity = await asyncio.wait_for(
                self.client.get_entity(channel_name),
                timeout=config.request_timeout
            )
            
            messages = []
            message_count = 0
            async for message in self.client.iter_messages(entity, limit=100):
                if shutdown_event.is_set():
                    break
                
                try:
                    if not hasattr(message, 'date') or not message.date:
                        continue
                        
                    media_info = await self.download_media(message, channel_name)
                    
                    message_data = {
                        'channel': channel_name,
                        'channel_title': getattr(entity, 'title', ''),
                        'message_id': message.id,
                        'text': safe_encode(message.text) if message.text else '[no text]',
                        'date': format_timestamp(message.date),
                        'views': getattr(message, 'views', None),
                        'forwards': getattr(message, 'forwards', None),
                        'replies': getattr(message.replies, 'replies', None) if message.replies else None,
                        'has_media': bool(message.media),
                        'media_type': media_info['media_type'] if media_info else None,
                        'media_size': media_info['media_size'] if media_info else None,
                        'message_hash': generate_file_hash(message.text or ""),
                    }
                    
                    messages.append(message_data)
                    self.all_messages.append(message_data)  # Add to combined collection
                    message_count += 1
                    
                    if message_count % 10 == 0:
                        logger.debug(f"Collected {message_count} messages from {channel_name}")
                    
                    await asyncio.sleep(config.rate_limit_delay)
                    
                except Exception as e:
                    logger.debug(f"Skipping message {message.id}: {str(e)}")
                    continue
            
            logger.info(f"Finished scraping {channel_name}: {len(messages)} messages")
            return messages
            
        except errors.ChannelPrivateError:
            logger.error(f"Channel {channel_name} is private")
        except errors.ChannelInvalidError:
            logger.error(f"Channel {channel_name} doesn't exist")
        except Exception as e:
            logger.error(f"Error scraping {channel_name}: {str(e)}")
        return []

    async def save_messages(self, channel_name: str, messages: List[Dict]) -> bool:
        if not messages:
            logger.warning(f"No messages to save for {channel_name}")
            return False
        
        safe_name = channel_name.replace('@', '').replace('/', '_')
        base_path = config.output_dir / safe_name
        
        try:
            # Save as JSON
            json_path = base_path.with_suffix('.json')
            async with aiofiles.open(json_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(messages, ensure_ascii=False, indent=2))
            logger.info(f"Saved {len(messages)} messages to {json_path}")
            
            # Save as Parquet
            df = pd.DataFrame(messages)
            parquet_path = base_path.with_suffix('.parquet')
            df.to_parquet(parquet_path, index=False)
            logger.info(f"Saved Parquet to {parquet_path}")
            
            # Save as CSV
            csv_path = base_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Saved CSV to {csv_path}")
            
            # Create ZIP if we have images
            if any(msg.get('has_media', False) for msg in messages):
                zip_path = await self.create_zip_archive(channel_name)
                logger.info(f"Saved images to ZIP: {zip_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data for {channel_name}: {str(e)}")
            return False

    async def save_combined_csv(self):
        """Save all messages from all channels to a single CSV file"""
        if not self.all_messages:
            logger.warning("No messages to save in combined CSV")
            return False
        
        try:
            combined_path = config.output_dir / "all_messages_combined.csv"
            df = pd.DataFrame(self.all_messages)
            
            # Reorder columns for better readability
            columns = [
                'channel', 'channel_title', 'message_id', 'date', 'text',
                'views', 'forwards', 'replies', 'has_media',
                'media_type', 'media_size', 'message_hash'
            ]
            df = df[[col for col in columns if col in df.columns]]
            
            df.to_csv(combined_path, index=False, encoding='utf-8')
            logger.info(f"Saved combined CSV to {combined_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save combined CSV: {str(e)}")
            return False

class TelegramScraperApp:
    def __init__(self):
        self.client = None
        self.scraper = None
        self.new_session_created = False

    async def initialize_client(self):
        """Initialize client with proper session handling"""
        try:
            # Initialize without session first if none available
            if not config.session_string:
                logger.warning("No valid session found, creating new one...")
                self.client = TelegramClient(
                    None,  # Will create new session
                    config.api_id,
                    config.api_hash,
                    request_retries=config.max_retries,
                    connection_retries=config.max_retries
                )
                self.new_session_created = True
            else:
                # Ensure proper padding for base64
                session_str = config.session_string
                if len(session_str) % 4 != 0:
                    session_str += '=' * (4 - len(session_str) % 4)
                
                self.client = TelegramClient(
                    StringSession(session_str),
                    config.api_id,
                    config.api_hash,
                    request_retries=config.max_retries,
                    connection_retries=config.max_retries
                )

            await self.client.connect()
            
            if not await self.client.is_user_authorized():
                await self.handle_login()
                
                # Save the new session if we created one
                if self.new_session_created:
                    session_str = self.client.session.save()
                    if session_str:  # Ensure we have a session string
                        with open('session.txt', 'w') as f:
                            f.write(session_str)
                        logger.info("New session created and saved to session.txt")
                    else:
                        logger.error("Failed to get session string after login")

            me = await self.client.get_me()
            logger.info(f"Successfully logged in as: {me.first_name}")
            self.scraper = TelegramScraper(self.client)
            
        except Exception as e:
            logger.error(f"Client initialization failed: {str(e)}")
            raise

    async def handle_login(self):
        """Handle the login process with phone verification"""
        try:
            await self.client.send_code_request(config.phone_number)
            code = input('Enter the verification code sent to your Telegram: ')
            await self.client.sign_in(config.phone_number, code)
        except errors.SessionPasswordNeededError:
            password = input('Enter your 2FA password: ')
            await self.client.sign_in(password=password)
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            raise

    async def scrape_all_channels(self):
        if not self.scraper:
            raise RuntimeError("Scraper not initialized")
        
        results = []
        for channel in config.channels:
            if shutdown_event.is_set():
                break
                
            messages = await self.scraper.scrape_channel(channel)
            if messages:
                await self.scraper.save_messages(channel, messages)
                results.extend(messages)
        
        if results:
            # Save combined results
            await self.scraper.save_combined_csv()
            await self.save_combined_results(results)

    async def save_combined_results(self, messages: List[Dict]):
        combined_path = config.output_dir / "all_messages_combined.parquet"
        try:
            df = pd.DataFrame(messages)
            df.to_parquet(combined_path, index=False)
            logger.info(f"Saved combined Parquet to {combined_path}")
            
            metadata = {
                'total_messages': len(messages),
                'channels': list({msg['channel'] for msg in messages}),
                'generated_at': datetime.now(config.timezone).isoformat()
            }
            
            metadata_path = config.output_dir / "metadata.json"
            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, indent=2))
                
        except Exception as e:
            logger.error(f"Failed to save combined results: {str(e)}")

    async def run(self):
        try:
            await self.initialize_client()
            await self.scrape_all_channels()
        except Exception as e:
            logger.error(f"Application error: {str(e)}", exc_info=True)
        finally:
            if self.client:
                await self.client.disconnect()

async def main():
    print("\n" + "="*50)
    print("EthioMart Telegram Scraper")
    print(f"Targeting {len(config.channels)} channels")
    print("="*50 + "\n")
    
    try:
        app = TelegramScraperApp()
        await app.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        print("\nScraping completed. Check these locations:")
        print(f"- Individual channel files: {config.output_dir.resolve()}")
        print(f"- Combined CSV: {(config.output_dir / 'all_messages_combined.csv').resolve()}")
        print(f"- Image files: {config.media_dir.resolve()}")
        print(f"- Log file: {Path('telethon_scraper.log').resolve()}")
        if app.new_session_created and app.client:
            print("\nNEW SESSION STRING (save this for future runs):")
            print(app.client.session.save())

if __name__ == '__main__':
    asyncio.run(main())