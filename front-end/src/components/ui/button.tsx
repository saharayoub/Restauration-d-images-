import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium transition-all duration-200 disabled:pointer-events-none disabled:opacity-40 cursor-pointer select-none",
  {
    variants: {
      variant: {
        default:
          "bg-[var(--accent)] text-[#0a0a0b] hover:bg-[var(--accent)]/90 font-semibold",
        outline:
          "border border-[var(--border-hi)] text-[var(--text)] hover:border-[var(--accent)] hover:text-[var(--accent)] bg-transparent",
        ghost:
          "text-[var(--muted)] hover:text-[var(--text)] hover:bg-[var(--border)] bg-transparent",
        danger:
          "bg-[var(--danger)]/10 border border-[var(--danger)]/30 text-[var(--danger)] hover:bg-[var(--danger)]/20",
      },
      size: {
        sm:   "h-8  px-3 text-xs rounded-[var(--radius)]",
        md:   "h-10 px-5 rounded-[var(--radius)]",
        lg:   "h-12 px-8 text-base rounded-[var(--radius)]",
        icon: "h-9 w-9 rounded-[var(--radius)]",
      },
    },
    defaultVariants: { variant: "default", size: "md" },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
