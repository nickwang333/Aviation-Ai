import * as React from "react";
import { cn } from "@/lib/utils";
import { Slot } from "@radix-ui/react-slot";

export interface NavbarItemProps extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  asChild?: boolean;
}

const NavbarItem = React.forwardRef<HTMLAnchorElement, NavbarItemProps>(
  ({ className, asChild, ...props }, ref) => {
    const Comp = asChild ? Slot : "a";
    return (
      <Comp
        ref={ref}
        className={cn("px-4 py-2 text-sm font-medium hover:underline", className)}
        {...props}
      />
    );
  }
);
NavbarItem.displayName = "NavbarItem";

export { NavbarItem };
