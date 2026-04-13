import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent } from 'react';
import { FaTrashAlt, FaEye, FaEyeSlash } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  isAutoCaptioning: boolean;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  isAutoCaptioning,
  children,
  className = '',
  onDelete = () => {},
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchCaption = async () => {
    if (isCaptionLoaded) return;
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    apiClient
      .post(`/api/caption/get`, { imgPath: imageUrl }, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        console.log('Caption fetched:', data);
        let captionText = '';
        if (typeof data === 'object' && data !== null) {
          // If data is an object (parsed JSON), convert it back to string
          // 兼容后端返回 JSON 的情况，保持前端编辑一致性
          captionText = JSON.stringify(data, null, 2);
        } else {
          // Otherwise ensure it's a string
          // 统一转为字符串，避免类型不一致导致 UI 异常
          captionText = String(data || '');
        }

        setCaption(captionText);
        setSavedCaption(captionText);
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching caption:', error);
      })
      .finally(() => {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      });
  };

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    apiClient
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption })
      .then(res => res.data)
      .then(data => {
        console.log('Caption saved:', data);
        setSavedCaption(trimmedCaption);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  // Only fetch caption when the component is both in viewport and visible
  useEffect(() => {
    if (inViewport && isVisible) {
      fetchCaption();
    }
  }, [inViewport, isVisible, isCaptionLoaded]);

  // Poll for caption updates every 5 seconds while auto-captioning
  useEffect(() => {
    if (!isAutoCaptioning || !inViewport || !isVisible) return;
    const interval = setInterval(() => {
      // Reset so fetchCaption will re-fetch
      setIsCaptionLoaded(false);
    }, 5000);
    return () => clearInterval(interval);
  }, [isAutoCaptioning, inViewport, isVisible]);

  useEffect(() => {
    // Create intersection observer to check viewport visibility
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInViewport(true);
          // Initialize isVisible to true when first coming into view
          if (!isVisible) {
            setIsVisible(true);
          }
        } else {
          setInViewport(false);
          // Cancel any in-flight caption fetch when scrolling away
          abortControllerRef.current?.abort();
        }
      },
      { threshold: 0.1 },
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  const toggleVisibility = (): void => {
    setIsVisible(prev => !prev);
    if (!isVisible && !isCaptionLoaded) {
      fetchCaption();
    }
  };

  const handleLoad = (): void => {
    setLoaded(true);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    // If Enter is pressed without Shift, prevent default behavior and save
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  const [showAudioPlayer, setShowAudioPlayer] = useState(true);

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  return (
    <div className={`flex flex-col ${className}`}>
      {/* Square image container */}
      <div
        ref={cardRef}
        className="relative w-full"
        style={{ paddingBottom: '100%' }} // Make it square
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md">
          {inViewport && isVisible && (
            <>
              {isItAVideo && (
                <video
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  className={`w-full h-full object-contain`}
                  autoPlay={false}
                  loop
                  muted
                  controls
                />
              )}
              {isItAudio && !showAudioPlayer && (
                <div
                  className="w-full h-full cursor-pointer flex items-center justify-center bg-gray-900"
                  onClick={() => setShowAudioPlayer(true)}
                >
                  <img
                    src={`/api/audio/art/${encodeURIComponent(imageUrl)}`}
                    alt={alt}
                    className="w-full h-full object-contain"
                    onError={e => {
                      (e.target as HTMLImageElement).style.display = 'none';
                    }}
                  />
                </div>
              )}
              {isItAudio && showAudioPlayer && (
                <AudioPlayer
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  title={imageUrl.replace(/^.*[\\/]/, '')}
                />
              )}
              {isItImage && (
                <img
                  src={`/api/img/${encodeURIComponent(imageUrl)}`}
                  alt={alt}
                  onLoad={handleLoad}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}
          {!isVisible && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded-t-lg">
              <span className="text-white text-lg"></span>
            </div>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          <div className="absolute top-1 right-1 flex space-x-2 z-10">
            <button
              className="bg-gray-800 rounded-full p-2"
              onClick={() => {
                openConfirm({
                  title: `删除${isItAVideo ? '视频' : '图片'}`,
                  message: `确定要删除这个${isItAVideo ? '视频' : '图片'}吗？该操作不可撤销。`,
                  type: 'warning',
                  confirmText: '删除',
                  onConfirm: () => {
                    apiClient
                      .post('/api/img/delete', { imgPath: imageUrl })
                      .then(() => {
                        console.log('Image deleted:', imageUrl);
                        onDelete();
                      })
                      .catch(error => {
                        console.error('Error deleting image:', error);
                      });
                  },
                });
              }}
            >
              <FaTrashAlt />
            </button>
          </div>
        </div>
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {inViewport && isVisible && (isCaptionLoaded || caption) && (
          <form
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            <textarea
              className={classNames("w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none", {
                'opacity-50 cursor-not-allowed': isAutoCaptioning,
              })}
              value={caption}
              rows={3}
              readOnly={isAutoCaptioning}
              onChange={e => setCaption(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        )}
        {(!inViewport || !isVisible) && isCaptionLoaded && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            {isVisible ? '滚动到可视区域以编辑' : '显示内容以编辑'}
          </div>
        )}
        {!isCaptionLoaded && !caption && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">加载标签中...</div>
        )}
      </div>
    </div>
  );
};

export default DatasetImageCard;
