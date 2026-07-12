/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ChartDataPoint, Track } from '../types';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { Activity, Zap, Cpu, Compass } from 'lucide-react';

interface StatsPanelProps {
  tracks: Track[];
  historyData: ChartDataPoint[];
  fps: number;
  latencyMs: number;
  totalLoggedTargetsCount: number;
  colorTheme: 'purple' | 'cyan' | 'pink' | 'amber';
}

export default function StatsPanel({
  tracks,
  historyData,
  fps,
  latencyMs,
  totalLoggedTargetsCount,
  colorTheme
}: StatsPanelProps) {
  // Theme highlights mapping
  const neonPalette = {
    purple: {
      primary: '#a855f7',
      secondary: '#6366f1',
      glow: 'rgba(168,85,247,0.4)',
      bgGrad: 'from-purple-500/20 to-indigo-500/5'
    },
    cyan: {
      primary: '#06b6d4',
      secondary: '#10b981',
      glow: 'rgba(6,182,212,0.4)',
      bgGrad: 'from-cyan-500/20 to-teal-500/5'
    },
    pink: {
      primary: '#f43f5e',
      secondary: '#ec4899',
      glow: 'rgba(244,63,94,0.4)',
      bgGrad: 'from-rose-500/20 to-pink-500/5'
    },
    amber: {
      primary: '#f59e0b',
      secondary: '#ef4444',
      glow: 'rgba(245,158,11,0.4)',
      bgGrad: 'from-amber-500/20 to-orange-500/5'
    }
  }[colorTheme];

  // Group current active tracks by category
  const activeCountAndCategories = React.useMemo(() => {
    let person = 0;
    let vehicle = 0;
    let baggage = 0;
    let sports = 0;
    let other = 0;

    tracks.forEach(tr => {
      const cls = tr.class.toLowerCase();
      if (cls === 'person') {
        person++;
      } else if (['car', 'bicycle', 'motorcycle', 'truck', 'bus', 'train'].includes(cls)) {
        vehicle++;
      } else if (['backpack', 'handbag', 'suitcase', 'umbrella'].includes(cls)) {
        baggage++;
      } else if (['sports ball', 'frisbee', 'skis'].includes(cls)) {
        sports++;
      } else {
        other++;
      }
    });

    return [
      { name: 'Persons', value: person, color: '#6366f1' },
      { name: 'Vehicles', value: vehicle, color: '#06b6d4' },
      { name: 'Baggage', value: baggage, color: '#f43f5e' },
      { name: 'Sports', value: sports, color: '#f59e0b' },
      { name: 'Others', value: other, color: '#10b981' }
    ].filter(item => item.value > 0 || tracks.length === 0);
  }, [tracks]);

  // Compute stats metrics
  const avgVelocity = React.useMemo(() => {
    if (tracks.length === 0) return 0;
    const sum = tracks.reduce((acc, tr) => acc + (tr.speed || 0), 0);
    return Number((sum / tracks.length).toFixed(1));
  }, [tracks]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 w-full">
      {/* CARD 1: DYNAMIC TARGETS */}
      <div className="relative bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-4 shadow-[0_4px_25px_rgba(0,0,0,0.3)] overflow-hidden flex flex-col justify-between min-h-[110px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">Active Targets</p>
            <h3 className="text-3xl font-extrabold text-white mt-1 tracking-tight">
              {tracks.length} <span className="text-xs font-normal text-slate-400">tracked</span>
            </h3>
          </div>
          <div className="p-2.5 rounded-xl bg-purple-950/30 border border-purple-900/30 flex items-center justify-center">
            <Compass className="w-5 h-5 text-purple-400 animate-spin-slow" />
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 text-[10px] font-mono text-emerald-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>Cumulative Unique Sensors locked: {totalLoggedTargetsCount}</span>
        </div>
        {/* Glow effect */}
        <div className="absolute right-0 bottom-0 w-16 h-16 bg-purple-500/5 blur-xl rounded-full" />
      </div>

      {/* CARD 2: FPS COUNTER */}
      <div className="relative bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-4 shadow-[0_4px_25px_rgba(0,0,0,0.3)] overflow-hidden flex flex-col justify-between min-h-[110px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">CV Engine Processing Rate</p>
            <h3 className="text-3xl font-extrabold text-white mt-1 tracking-tight">
              {fps.toFixed(1)} <span className="text-xs font-normal text-slate-500">FPS</span>
            </h3>
          </div>
          <div className="p-2.5 rounded-xl bg-cyan-950/30 border border-cyan-900/30 flex items-center justify-center">
            <Activity className="w-5 h-5 text-cyan-400" />
          </div>
        </div>
        <div className="mt-3 text-[10px] font-mono text-slate-400">
          Sync cycle performance: <span className="text-cyan-400">{fps > 25 ? 'Smooth' : fps > 12 ? 'Nominal' : 'Loading/Idle'}</span>
        </div>
        <div className="absolute right-0 bottom-0 w-16 h-16 bg-cyan-500/5 blur-xl rounded-full" />
      </div>

      {/* CARD 3: LATENCY ms */}
      <div className="relative bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-4 shadow-[0_4px_25px_rgba(0,0,0,0.3)] overflow-hidden flex flex-col justify-between min-h-[110px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">Detection Pipeline Latency</p>
            <h3 className="text-3xl font-extrabold text-white mt-1 tracking-tight">
              {latencyMs} <span className="text-xs font-normal text-slate-500">ms</span>
            </h3>
          </div>
          <div className="p-2.5 rounded-xl bg-pink-950/30 border border-pink-900/30 flex items-center justify-center">
            <Cpu className="w-5 h-5 text-pink-400" />
          </div>
        </div>
        <div className="mt-3 text-[10px] font-mono text-slate-400">
          Inference & draw overload: <span className="text-pink-400">{latencyMs < 30 ? 'Turbo' : latencyMs < 80 ? 'Standard' : 'Heavy'}</span>
        </div>
        <div className="absolute right-0 bottom-0 w-16 h-16 bg-pink-500/5 blur-xl rounded-full" />
      </div>

      {/* CARD 4: AVG VELOCITY */}
      <div className="relative bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-4 shadow-[0_4px_25px_rgba(0,0,0,0.3)] overflow-hidden flex flex-col justify-between min-h-[110px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-[10px] font-mono tracking-wider text-slate-400 uppercase">Estimated Mean Velocity</p>
            <h3 className="text-3xl font-extrabold text-white mt-1 tracking-tight">
              {avgVelocity} <span className="text-xs font-normal text-slate-500">px/f</span>
            </h3>
          </div>
          <div className="p-2.5 rounded-xl bg-amber-950/30 border border-amber-900/30 flex items-center justify-center">
            <Zap className="w-5 h-5 text-amber-400" />
          </div>
        </div>
        <div className="mt-3 text-[10px] font-mono text-amber-400">
          Dynamic Kinetic Activity factor: <span className="text-slate-400">{tracks.length > 0 ? 'Flowing' : 'No motion'}</span>
        </div>
        <div className="absolute right-0 bottom-0 w-16 h-16 bg-amber-500/5 blur-xl rounded-full" />
      </div>

      {/* GRID BREAK: CHARTS FOR IN-DEPTH STATISTICS */}
      <div className="md:col-span-2 bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-5 shadow-[0_4px_25px_rgba(0,0,0,0.35)] flex flex-col h-[260px]">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-xs font-mono font-bold tracking-wider text-purple-300 uppercase">
            Timeline Statistics (Target Over Time)
          </h4>
          <span className="text-[10px] font-mono text-slate-500">Rolling window (30 periods)</span>
        </div>
        <div className="flex-1 w-full min-h-0 [content-visibility:auto]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historyData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="glowColorGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={neonPalette.primary} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={neonPalette.primary} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1b4b/30" vertical={false} />
              <XAxis dataKey="time" stroke="#475569" fontSize={10} tickLine={false} />
              <YAxis stroke="#475569" fontSize={10} tickLine={false} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#020617',
                  borderColor: '#3b0764',
                  borderRadius: '12px',
                  fontSize: '11px',
                  fontFamily: 'monospace',
                  color: '#f8fafc'
                }}
              />
              <Area
                type="monotone"
                dataKey="activeCount"
                name="Total Tracked"
                stroke={neonPalette.primary}
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#glowColorGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="md:col-span-2 bg-slate-950/60 backdrop-blur-md rounded-2xl border border-purple-950/20 p-5 shadow-[0_4px_25px_rgba(0,0,0,0.35)] flex flex-col h-[260px]">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-xs font-mono font-bold tracking-wider text-purple-300 uppercase">
            Categorical Class Composition
          </h4>
          <span className="text-[10px] font-mono text-slate-500">Active distribution</span>
        </div>
        {tracks.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-slate-600 font-mono text-xs gap-2">
            <Compass className="w-10 h-10 text-slate-800 animate-spin-slow" />
            <span>Awaiting active target taxonomy logs...</span>
          </div>
        ) : (
          <div className="flex-1 w-full min-h-0 [content-visibility:auto]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={activeCountAndCategories} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e1b4b/30" vertical={false} />
                <XAxis dataKey="name" stroke="#475569" fontSize={10} tickLine={false} />
                <YAxis stroke="#475569" fontSize={10} tickLine={false} allowDecimals={false} />
                <Tooltip
                  cursor={{ fill: 'rgba(168,85,247,0.05)' }}
                  contentStyle={{
                    backgroundColor: '#020617',
                    borderColor: '#3b0764',
                    borderRadius: '12px',
                    fontSize: '11px',
                    fontFamily: 'monospace',
                    color: '#f8fafc'
                  }}
                />
                <Bar dataKey="value" name="Detected" radius={[6, 6, 0, 0]} maxBarSize={35}>
                  {activeCountAndCategories.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color || neonPalette.primary} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}
