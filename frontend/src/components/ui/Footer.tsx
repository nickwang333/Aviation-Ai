import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const footerVariants = cva(
  "w-full py-4 px-6 flex items-center justify-between border-t",
  {
    variants: {
      variant: {
        default: "bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-200",
        dark: "bg-gray-900 text-gray-200 border-gray-700",
        minimal: "bg-transparent border-none",
      },
      alignment: {
        left: "justify-start",
        center: "justify-center",
        right: "justify-end",
      },
    },
    defaultVariants: {
      variant: "default",
      alignment: "center",
    },
  }
);

export interface FooterProps extends React.HTMLAttributes<HTMLElement>, VariantProps<typeof footerVariants> {
    as?: React.ElementType;
    logo?: React.ReactNode;
    links?: { label: string; href: string }[];
    socialIcons?: React.ReactNode;
  }
  
  const Footer = React.forwardRef<HTMLElement, FooterProps>(
    ({ as: Comp = "footer", className, variant, alignment, logo, links, socialIcons, ...props }, ref) => {
      return (
        <Comp ref={ref as React.Ref<HTMLElement>} className={cn(footerVariants({ variant, alignment }), className)} {...props}>
          {logo && <div className="flex items-center">{logo}</div>}
          {links && (
            <nav className="flex space-x-4">
              {links.map((link, index) => (
                <a key={index} href={link.href} className="text-sm hover:underline">
                  {link.label}
                </a>
              ))}
            </nav>
          )}
          {socialIcons && <div className="flex space-x-3">{socialIcons}</div>}
        </Comp>
      );
    }
  );
  