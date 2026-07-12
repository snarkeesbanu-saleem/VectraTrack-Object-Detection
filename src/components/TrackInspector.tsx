/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Track } from '../types';
import { Crosshair, Navigation, History, Shield, Compass } from 'lucide-react';

interface TrackInspectorProps {
  selectedTrack: Track | null;
  onDeselect: () => void;
  colorTheme: 'purple' | 'cyan' | 'pink' | 'amber';
}

export default function TrackInspector({ selectedTrack, onDeselect, colorTheme }: TrackInspectorProps) {
  const neonColor = {
    purple: 'text-purple-400 border-purple-950/40 bg-purple-950/25 shadow-purple-500/10',
    cyan: 'text-cyan-400 border-cyan-950/40 bg-cyan-950/25 shadow-cyan-500/10',
    pink: 'text-pink-400 border-pink-950/40 bg-pink-950/25 shadow-pink-500/10',
    amber: 'text-amber-400 border-amber-950/40 bg-amber-950/25 shadow-amber-500/10'
  }[colorTheme];

  const badgeTheme = {
    purple: 'bg-purple-900/30 text-purple-300 border-purple-800/40',
    cyan: 'bg-cyan-900/30 text-cyan-300 border-cyan-800/40',
    pink: 'bg-rose-900/30 text-rose-300 border-rose-800/40',
    amber: 'bg-amber-900/30 text-amber-300 border-amber-800/40'
  }[colorTheme];

  if (!selectedTrack) {
    return (
      <div className="flex flex-col items-center justify-center h-full bg-slate-950/50 backdrop-blur-md rounded-2xl border border-purple-950/15 p-6 text-center text-slate-500 gap-3 min-h-[340px]">
        <div className="p-4 rounded-full bg-slate-900 border border-purple-950/20 text-slate-700 animate-pulse">
          <Crosshair className="w-8 h-8" />
        </div>
        <div>
          <h4 className="text-sm font-semibold text-slate-400 font-sans">No Target Segment Selected</h4>
          <p className="text-xs text-slate-600 mt-1 max-w-[240px] mx-auto font-sans">
            Tap directly on any bounding box or log entry to isolate focal sensor telemetry.
          </p>
        </div>
      </div>
    );
  }

  // Calculate motion heading angle in degrees (0 is right, rotating clockwise)
  const [dx, dy] = selectedTrack.velocity;
  const headingRad = Math.atan2(dy, dx);
  const headingDeg = Math.round((headingRad * 180) / Math.PI);
  // Normalize to 0-360 degrees
  const normalizedHeading = headingDeg < 0 ? headingDeg + 360 : headingDeg;

  // Determine compass direct label
  const getCompassDir = (deg: number) => {
    if (deg >= 337.5 || deg < 22.5) return 'East ➡️';
    if (deg >= 22.5 && deg < 67.5) return 'South-East ↘️';
    if (deg >= 67.5 && deg < 112.5) return 'South ⬇️';
    if (deg >= 112.5 && deg < 157.5) return 'South-West ↙️';
    if (deg >= 157.5 && deg < 202.5) return 'West ⬅️';
    if (deg >= 202.5 && deg < 247.5) return 'North-West ↖️';
    if (deg >= 247.5 && deg < 292.5) return 'North ⬆️';
    return 'North-East ↗️';
  };

  return (
    <div className={`flex flex-col bg-slate-950/85 backdrop-blur-md rounded-2xl border ${neonColor} p-5 shadow-[0_0_30px_rgba(0,0,0,0.5)] min-h-[340px] text-slate-200 transition-all duration-300`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-purple-950/30 pb-3 mb-4">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4.5 h-4.5 text-purple-400 animate-spin-slow" />
          <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-purple-300">
            Focal Segment Telemetry Inspector
          </h3>
        </div>
        <button
          onClick={onDeselect}
          className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 hover:text-white transition-colors"
        >
          RELEASE LOCK
        </button>
      </div>

      {/* Primary specs row */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/40">
          <label className="text-[9px] font-mono text-slate-500 uppercase">Target Signature ID</label>
          <p className="text-xl font-extrabold font-mono text-white mt-1">
            #{selectedTrack.id}
          </p>
        </div>
        <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/40">
          <label className="text-[9px] font-mono text-slate-500 uppercase">Object Taxonomy Category</label>
          <span className={`inline-block text-xs font-semibold px-2 py-1.5 rounded border uppercase tracking-wider mt-1.5 ${badgeTheme}`}>
            🏷️ {selectedTrack.class}
          </span>
        </div>
      </div>

      {/* Target status details */}
      <div className="space-y-3 flex-1">
        {/* Core numbers */}
        <div className="grid grid-cols-2 gap-3 text-xs font-mono">
          <div className="flex justify-between border-b border-slate-900 pb-1">
            <span className="text-slate-500">Detector Strength:</span>
            <span className="text-emerald-400 font-bold">{(selectedTrack.score * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 pb-1">
            <span className="text-slate-500">Vector Speed:</span>
            <span className="text-cyan-400 font-bold">{selectedTrack.speed} px/f</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 pb-1">
            <span className="text-slate-500">Lifetime Frame Age:</span>
            <span className="text-slate-350">{selectedTrack.age} frames</span>
          </div>
          <div className="flex justify-between border-b border-slate-900 pb-1">
            <span className="text-slate-500">Engine Match State:</span>
            <span className={selectedTrack.isActive ? 'text-emerald-400 font-bold' : 'text-amber-400 font-bold animate-pulse'}>
              {selectedTrack.isActive ? '📡 LOCKED ON' : '🔄 ESTIMATING'}
            </span>
          </div>
        </div>

        {/* Bounding box position coords */}
        <div className="bg-slate-950/60 p-3 rounded-lg border border-purple-950/20 font-mono text-[10px] space-y-1.5 text-slate-400">
          <div className="text-purple-400 font-bold uppercase tracking-wider mb-1 flex items-center gap-1">
            <Shield className="w-3.5 h-3.5" /> Coord Translation (Px Space)
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <div className="flex justify-between">
              <span>Anchor X0:</span> <span className="text-white">{Math.round(selectedTrack.bbox[0])}px</span>
            </div>
            <div className="flex justify-between">
              <span>Anchor Y0:</span> <span className="text-white">{Math.round(selectedTrack.bbox[1])}px</span>
            </div>
            <div className="flex justify-between">
              <span>Bbox Width:</span> <span className="text-white">{Math.round(selectedTrack.bbox[2])}px</span>
            </div>
            <div className="flex justify-between">
              <span>Bbox Height:</span> <span className="text-white">{Math.round(selectedTrack.bbox[3])}px</span>
            </div>
          </div>
        </div>

        {/* Linear movement vector (dx dy orientation) */}
        <div className="bg-slate-950/60 p-3 rounded-lg border border-purple-950/20 font-mono text-[10px] space-y-2">
          <div className="text-purple-400 font-bold uppercase tracking-wider flex items-center gap-1">
            <Navigation className="w-3.5 h-3.5" /> Kinematic Heading Vector
          </div>
          <div className="grid grid-cols-2 gap-2 text-slate-400">
            <div className="flex justify-between">
              <span>Shift dX:</span>
              <span className="text-cyan-300 font-bold">{dx.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span>Shift dY:</span>
              <span className="text-cyan-300 font-bold">{dy.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span>Direction:</span>
              <span className="text-white font-bold">{normalizedHeading}°</span>
            </div>
            <div className="flex justify-between">
              <span>Compass:</span>
              <span className="text-emerald-400 font-bold truncate">{getCompassDir(normalizedHeading)}</span>
            </div>
          </div>
        </div>

        {/* Trail node counts history summary */}
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 pt-1.5">
          <History className="w-3.5 h-3.5" />
          <span>Track history holds {selectedTrack.history.length} path nodes. Coordinates mapped.</span>
        </div>
      </div>
    </div>
  );
}
