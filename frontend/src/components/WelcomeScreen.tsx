export default function WelcomeScreen() {
    return (
        <div className="flex-1 flex flex-col items-center justify-center px-4 animate-fade-in">
            {/* Arc Reactor Logo */}
            <div className="relative w-24 h-24 mb-8">
                {/* Outer ring */}
                <div className="absolute inset-0 rounded-full border-2 border-arc-blue/30 animate-arc-spin" />
                <div className="absolute inset-1 rounded-full border border-arc-blue/20" />
                {/* Middle ring with segments */}
                <div className="absolute inset-3 rounded-full border-2 border-arc-blue/50">
                    <div className="absolute inset-0 rounded-full"
                        style={{
                            background: 'conic-gradient(from 0deg, transparent 0deg, rgba(79,195,247,0.3) 30deg, transparent 60deg, rgba(79,195,247,0.3) 120deg, transparent 150deg, rgba(79,195,247,0.3) 210deg, transparent 240deg, rgba(79,195,247,0.3) 300deg, transparent 330deg)'
                        }}
                    />
                </div>
                {/* Inner core */}
                <div className="absolute inset-6 rounded-full bg-arc-blue/20 animate-arc-pulse" />
                <div className="absolute inset-7 rounded-full bg-arc-blue/40" />
                <div className="absolute inset-8 rounded-full bg-arc-blue shadow-[0_0_20px_rgba(79,195,247,0.6),0_0_40px_rgba(79,195,247,0.3)]" />
            </div>

            {/* Title */}
            <h1 className="font-display text-3xl md:text-4xl font-bold text-iron-text mb-2 animate-glow tracking-wider">
                J.A.R.V.I.S.
            </h1>
            <p className="text-iron-text-secondary text-base mb-2">
                Just A Rather Very Intelligent System
            </p>
            <p className="text-iron-text-muted text-sm mb-10 max-w-md text-center">
                Your AI assistant powered by a language model built from scratch.
                Ask me anything about machine learning, transformers, or general knowledge.
            </p>

            {/* Bottom decoration */}
            <div className="mt-12 flex items-center gap-3">
                <div className="h-px w-12 bg-gradient-to-r from-transparent to-iron-border" />
                <div className="flex items-center gap-1.5">
                    <div className="w-1 h-1 rounded-full bg-iron-red" />
                    <div className="w-1 h-1 rounded-full bg-iron-gold" />
                    <div className="w-1 h-1 rounded-full bg-arc-blue" />
                </div>
                <div className="h-px w-12 bg-gradient-to-l from-transparent to-iron-border" />
            </div>
        </div>
    );
}
