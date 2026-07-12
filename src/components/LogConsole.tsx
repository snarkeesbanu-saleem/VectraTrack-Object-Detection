/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef } from 'react';
import { SystemEvent } from '../types';
import { Terminal, Trash2, ShieldAlert } from 'lucide-react';

interface LogConsoleProps {
  events: SystemEvent[];
  onClearEvents: () => void;
}

export default function LogConsole({ events, onClearEvents }: LogConsoleProps) {
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [events]);

  const getEventColors = (type: SystemEvent['type']) => {
    switch (type) {
      case 'success':
        return { text: 'text-emerald-400', badge: 'bg-emerald-950/40 text-emerald-300 border-emerald-900/50' };
      case 'warning':
        return { text: 'text-amber-400', badge: 'bg-amber-950/40 text-amber-300 border-amber-900/50' };
      case 'alert':
        return { text: 'text-rose-400', badge: 'bg-rose-950/40 text-rose-300 border-rose-900/50 animate-pulse' };
      default:
        return { text: 'text-cyan-400', badge: 'bg-cyan-950/40 text-cyan-300 border-cyan-900/50' };
    }
  };

  return (
    <div className="flex flex-col bg-slate-950/80 rounded-2xl border border-purple-950/30 shadow-inner h-[280px]">
      {/* Console Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-950 border-b border-purple-950/30 rounded-t-2xl">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-purple-400 animate-pulse" />
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-purple-300">
            System Event Log Stream
          </span>
          <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-purple-950/50 text-purple-400 border border-purple-900/30">
            buffer: {events.length}/100
          </span>
        </div>
        
        <button
          onClick={onClearEvents}
          disabled={events.length === 0}
          className="text-slate-500 hover:text-rose-400 disabled:opacity-30 disabled:hover:text-slate-500 transition-colors duration-200"
          title="Flush buffer"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Terminal View area */}
      <div
        ref={terminalRef}
        className="flex-1 p-4 overflow-y-auto font-mono text-[11px] leading-relaxed space-y-2.5 scrollbar-thin scrollbar-thumb-purple-950 bg-slate-950/90 rounded-b-2xl h-[220px]"
      >
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-slate-600 gap-2">
            <Terminal className="w-8 h-8 text-slate-800" />
            <span>Telemetry stream idle. Start tracking engine...</span>
          </div>
        ) : (
          events.map((evt) => {
            const colors = getEventColors(evt.type);
            return (
              <div
                key={evt.id}
                className="flex items-start gap-2.5 transition-all duration-300 hover:bg-slate-900/40 py-1 px-1.5 rounded"
              >
                {/* Timestamp */}
                <span className="text-slate-500 select-none flex-shrink-0">
                  [{evt.timestamp.split('T')[1]?.slice(0, 8) || evt.timestamp}]
                </span>

                {/* Event Type Badge */}
                <span className={`text-[9px] px-1.5 py-0.2 rounded border uppercase font-bold flex-shrink-0 tracking-tight ${colors.badge}`}>
                  {evt.type}
                </span>

                {/* Event Message */}
                <span className={`flex-1 break-words leading-tight ${colors.text}`}>
                  {evt.message}
                </span>
                
                {/* Alarm Icon for critical tracking alerts */}
                {evt.type === 'alert' && (
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-500 animate-bounce flex-shrink-0" />
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
