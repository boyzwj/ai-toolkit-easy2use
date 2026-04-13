"use client";

import Link from 'next/link';
import { Home, Settings, BrainCircuit, Images, Plus, Heart } from 'lucide-react';
import { FaDiscord, FaYoutube } from 'react-icons/fa6';
import { SiBilibili } from 'react-icons/si';
import { useEffect, useState } from 'react';
import classNames from 'classnames';
import ThemeToggle from './ThemeToggle';
import ThemeLogo from './ThemeLogo';

const Sidebar = () => {
  const navigation = [
    { name: '仪表盘', href: '/dashboard', icon: Home },
    { name: '新建任务', href: '/jobs/new', icon: Plus },
    { name: '训练队列', href: '/jobs', icon: BrainCircuit },
    { name: '数据集', href: '/datasets', icon: Images },
    { name: '设置', href: '/settings', icon: Settings },
  ];

  const socialsBoxClass =
    'flex flex-col items-center justify-center p-1 hover:bg-gray-800 rounded-lg transition-colors';
  const socialIconClass = 'w-5 h-5 text-gray-400 hover:text-white';

  const [isOpenOnMobile, setIsOpenOnMobile] = useState(false);

  const toggleCollapse = () => setIsOpenOnMobile(c => !c);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const handler = () => toggleCollapse();
    window.addEventListener('aitk:toggleSidebar', handler as EventListener);
    return () => window.removeEventListener('aitk:toggleSidebar', handler as EventListener);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const collapsed = window.innerWidth < 768 ? !isOpenOnMobile : false;
    window.dispatchEvent(new CustomEvent('aitk:sidebarState', { detail: { collapsed } }));
  }, [isOpenOnMobile]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onRequest = () => {
      const collapsed = window.innerWidth < 768 ? !isOpenOnMobile : false;
      window.dispatchEvent(new CustomEvent('aitk:sidebarState', { detail: { collapsed } }));
    };
    window.addEventListener('aitk:requestSidebarState', onRequest as EventListener);
    return () => window.removeEventListener('aitk:requestSidebarState', onRequest as EventListener);
  }, [isOpenOnMobile]);

  const AvatarOrHeart = () => {
    const [useHeartIcon, setUseHeartIcon] = useState(false);
    const [srcIndex, setSrcIndex] = useState(0);
    const candidates = [
      '/doc_workbox_avatar.jpg',
      '/doc_workbox_avatar.png',
      '/doc_workbox_avatar.jpeg',
      '/doc_workbox_avatar.webp',
      '/doc_workbox_avatar.svg',
      '/doc_workbox_avatar.avif',
    ];

    if (useHeartIcon) {
      return <Heart className="w-6 h-6 text-pink-400" aria-label="Doc_workBox 爱心" />;
    }

    return (
      <img
        src={candidates[srcIndex]}
        alt="Doc_workBox 头像"
        className="w-6 h-6 rounded object-cover"
        onError={() => {
          const next = srcIndex + 1;
          if (next < candidates.length) {
            setSrcIndex(next);
          } else {
            setUseHeartIcon(true);
          }
        }}
      />
    );
  };

  return (
    <div
      suppressHydrationWarning
      className={classNames(
        'flex flex-col bg-gray-900 text-gray-100 transition-all duration-300',
        isOpenOnMobile ? 'w-[200px] md:w-[240px]' : 'w-0 md:w-[240px] overflow-hidden',
      )}
    >
      <div className="px-4 md:px-3 py-3">
        <div className="grid grid-cols-[auto_1fr] items-center gap-3 md:flex md:items-center">
          <div className="row-span-2 md:row-span-1">
            <ThemeLogo />
          </div>
          <span className="font-bold uppercase md:ml-0">OSTRIS</span>
          <div className="uppercase text-gray-300 text-sm whitespace-nowrap md:ml-0">AI-TOOLKIT-E2U</div>
        </div>
      </div>
      <nav className="flex-1">
        <ul className="px-2 py-4 space-y-2">
          {navigation.map(item => (
            <li key={item.name}>
              <Link
                href={item.href}
                className="flex items-center px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
                onClick={() => {
                  if (typeof window !== 'undefined' && window.innerWidth < 768) {
                    setIsOpenOnMobile(false);
                  }
                }}
              >
                <item.icon className="w-5 h-5 mr-3" />
                {item.name}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <div className="flex items-center space-x-2 px-4 py-3">
        <div className="min-w-[26px] min-h-[26px]">
          <AvatarOrHeart />
        </div>
        <div className="text-gray-500 text-sm mb-2 flex-1 pt-2 pl-0">由Doc_workBox汉化</div>
      </div>

      <div className="px-1 py-1 border-t border-gray-800">
        <div className="grid grid-cols-4 gap-4">
          <a href="https://discord.gg/VXmU2f5WEU" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaDiscord className={socialIconClass} />
          </a>
          <a href="https://www.youtube.com/@Doc_workBox" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <FaYoutube className={socialIconClass} />
          </a>
          <a href="https://space.bilibili.com/12710942" target="_blank" rel="noreferrer" className={socialsBoxClass}>
            <SiBilibili className={socialIconClass} />
          </a>
          <ThemeToggle />
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
