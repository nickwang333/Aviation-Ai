import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { Slot } from "@radix-ui/react-slot";

const navbarVariants = cva(
  "w-full flex items-center justify-between px-6 py-4 shadow-sm border-b transition-all",
  {
    variants: {
      variant: {
        default: "bg-white text-gray-900 dark:bg-gray-900 dark:text-white",
        transparent: "bg-transparent text-white",
        primary: "bg-primary text-white",
        secondary: "bg-secondary text-gray-900",
      },
      align: {
        left: "justify-start",
        center: "justify-center",
        right: "justify-end",
        between: "justify-between",
      },
    },
    defaultVariants: {
      variant: "default",
      align: "between",
    },
  }
);

export interface NavbarProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof navbarVariants> {
  asChild?: boolean;
}

const Navbar = React.forwardRef<HTMLElement, NavbarProps>(
  ({ className, variant, align, asChild, ...props }, ref) => {
    const Comp = asChild ? Slot : "nav";
    return (
      <Comp ref={ref} className={cn(navbarVariants({ variant, align }), className)} {...props} />
    );
  }
);
Navbar.displayName = "Navbar";

export { Navbar };
